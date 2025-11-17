import os
import time
import requests
import pandas as pd
from pathlib import Path
from pyproj import Transformer
import chardet
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

"""
Geocode addresses using Digitransit API. Coordinates are output in GK25FIN (EPSG:3879).
OPTIMIZED VERSION with parallel processing and caching.

Steps:
1. Set INPUT_CSV to your source CSV file path
2. Set ADDRESS_FIELDS = ["Street", "City", ...]
3. Add your Digitransit API_KEY
4. Adjust NUM_THREADS for parallel processing (default: 4)
5. Set LIMIT > 0 to test, 0 to process all rows

Output:
- Output is saved as 'Geocode/geocode_<inputfilename>.csv' in the same folder as input CSV.
- Adds columns: 'x_3879', 'y_3879' to the end of the CSV

Dependencies: pandas, requests, pyproj, chardet, tqdm
"""

# ============================================================================
# CONFIGURATION - EDIT HERE
# ============================================================================

INPUT_CSV = r"L:/4 YTET/10 Kaupunkisuunnittelu/03 Yleiskaava/3 Tutkimukset ja selvitykset/40 Asukaskyselyt/Koulu- ja päiväkotivihreä kysely/Paikkatiedot/Geokoodaus/aakkosnumeeriset.csv"

ADDRESS_FIELDS = ["s2k3: Yksikön osoite:", "Kaupunki"]

API_KEY = "8b2808e2bbd64c7694b488d536f70bc5"

# For testing: Limit the number of addresses to geocode (from the start of the CSV). Use 0 to process all rows
LIMIT = 0

# PERFORMANCE SETTINGS
# Number of parallel threads (increase for faster processing, but be respectful to API)
# Recommended: 2-4 for Digitransit (they may rate limit at higher values)
NUM_THREADS = 4

# Request delay between API calls PER THREAD (in seconds)
# Total throughput = NUM_THREADS * (1/REQUEST_DELAY) requests/sec
REQUEST_DELAY = 0.05

# Cache results to disk (speeds up re-runs on same data)
USE_CACHE = True
CACHE_FILE = "geocode_cache.json"

# ============================================================================
# FUNCTIONS
# ============================================================================

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_success(message):
    """Print a success message."""
    print(f"✓ {message}")


def print_warning(message):
    """Print a warning message."""
    print(f"⚠ {message}")


def print_error(message):
    """Print an error message."""
    print(f"✗ {message}")


def load_cache():
    """Load geocoding cache from disk."""
    if USE_CACHE and Path(CACHE_FILE).exists():
        try:
            import json
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print_success(f"Loaded cache with {len(cache)} entries")
            return cache
        except Exception as e:
            print_warning(f"Could not load cache: {e}")
    return {}


def save_cache(cache):
    """Save geocoding cache to disk."""
    if USE_CACHE:
        try:
            import json
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print_warning(f"Could not save cache: {e}")


def detect_file_encoding(file_path, sample_size=100000):
    """Detect the encoding of a file using chardet."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
        
        detection = chardet.detect(raw_data)
        return detection.get('encoding'), detection.get('confidence', 0)
    except Exception as e:
        print_error(f"Could not detect encoding: {e}")
        return None, 0


def attempt_read_csv(file_path, sep=',', encodings=None):
    """Attempt to read CSV with multiple encoding fallbacks."""
    if encodings is None:
        encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin1', 'iso-8859-1']
    
    print_header("Reading CSV File")
    print(f"File path: {file_path}")
    
    detected_enc, confidence = detect_file_encoding(file_path)
    print(f"Chardet detection: {detected_enc} (confidence: {confidence:.1%})")
    
    if detected_enc and detected_enc.lower() not in [e.lower() for e in encodings]:
        encodings = [detected_enc] + encodings
    
    for enc in encodings:
        try:
            print(f"  Trying encoding: {enc}...", end=" ", flush=True)
            df = pd.read_csv(file_path, sep=sep, encoding=enc, engine='python', on_bad_lines='warn')
            print(f"✓ SUCCESS")
            print_success(f"File read successfully with encoding: {enc} (shape: {df.shape})")
            return df, enc
        except UnicodeDecodeError:
            print("✗")
        except Exception as e:
            print("✗")
    
    return None, None


def validate_address_fields(df, address_fields):
    """Validate that all required address fields exist in the DataFrame."""
    missing_fields = [field for field in address_fields if field not in df.columns]
    
    if not missing_fields:
        return True
    
    print_error("Missing required address fields!")
    print(f"\nMissing fields:")
    for field in missing_fields:
        print(f"  - {field}")
    
    print(f"\nAvailable columns in CSV ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n" + "!"*70)
    print("SOLUTION:")
    print("!"*70)
    print("Update ADDRESS_FIELDS in the script to match the actual column names.")
    
    return False


def detect_csv_delimiter(file_path):
    """Attempt to detect the CSV delimiter by checking raw content."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            first_line = f.readline()
        
        comma_count = first_line.count(',')
        semicolon_count = first_line.count(';')
        tab_count = first_line.count('\t')
        
        delimiters = {',': comma_count, ';': semicolon_count, '\t': tab_count}
        most_likely = max(delimiters, key=delimiters.get)
        print(f"Detected delimiter: '{most_likely}' (commas: {comma_count}, semicolons: {semicolon_count}, tabs: {tab_count})")
        return most_likely
    except Exception as e:
        print_error(f"Could not detect delimiter: {e}")
        return ','


def clean_excel_quote(s):
    """Remove Excel formula quotes from a string."""
    s = str(s or '').strip()
    if s.startswith('="') and s.endswith('"'):
        return s[2:-1]
    return s


def geocode_address_worker(idx, row, address_fields, params_template, transformer, request_delay, cache, cache_lock):
    """
    Worker function for parallel geocoding.
    
    Returns:
        tuple: (idx, lon_3879, lat_3879, error_message or None)
    """
    address = " ".join(str(row.get(field, '') or '') for field in address_fields).strip()
    
    if not address or address.strip() == '':
        return idx, None, None, "Empty address"
    
    # Check cache first
    with cache_lock:
        if address in cache:
            cached = cache[address]
            if cached.get('error'):
                return idx, None, None, f"Cached error: {cached['error']}"
            return idx, cached['x'], cached['y'], None
    
    try:
        time.sleep(request_delay)
        
        params = params_template.copy()
        params["text"] = address
        
        response = requests.get(
            "https://api.digitransit.fi/geocoding/v1/search",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        features = data.get('features', [])
        if features:
            coords = features[0]['geometry']['coordinates']
            lon, lat = coords
            lon_3879, lat_3879 = transformer.transform(lon, lat)
            
            # Cache result
            with cache_lock:
                cache[address] = {'x': lon_3879, 'y': lat_3879}
            
            return idx, lon_3879, lat_3879, None
        else:
            error_msg = f"No match found"
            with cache_lock:
                cache[address] = {'error': error_msg}
            return idx, None, None, error_msg

    except requests.exceptions.RequestException as e:
        error_msg = f"API error: {str(e)[:50]}"
        return idx, None, None, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)[:50]}"
        return idx, None, None, error_msg


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    start_time = time.time()
    
    # Check if file exists
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print_error(f"Input file not found: {INPUT_CSV}")
        sys.exit(1)

    file_size_kb = input_path.stat().st_size / 1024
    print_header("GEOCODING SCRIPT (OPTIMIZED)")
    print(f"Input file: {input_path.name}")
    print(f"File size: {file_size_kb:.1f} KB")
    print(f"Current time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Performance settings:")
    print(f"  Parallel threads: {NUM_THREADS}")
    print(f"  Request delay: {REQUEST_DELAY}s per thread")
    print(f"  Expected throughput: {NUM_THREADS / REQUEST_DELAY:.1f} requests/sec")
    print(f"  Caching: {'Enabled' if USE_CACHE else 'Disabled'}")

    # Detect delimiter
    detected_delim = detect_csv_delimiter(INPUT_CSV)

    # Read input with encoding detection
    df, used_encoding = attempt_read_csv(INPUT_CSV, sep=detected_delim)

    if df is None:
        print_header("ENCODING ERROR")
        print_error("Could not read the CSV with any standard encoding.")
        sys.exit(1)

    # Validate address fields
    print_header("Validating Configuration")
    if not validate_address_fields(df, ADDRESS_FIELDS):
        print_error("Cannot proceed without valid address fields.")
        sys.exit(1)

    print_success("All address fields are valid!")
    print(f"Fields to use: {ADDRESS_FIELDS}\n")

    # Coordinate transformer: WGS84 -> EPSG:3879
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3879", always_xy=True)

    # Clean all address fields in the DataFrame
    print_header("Preparing Data")
    for field in ADDRESS_FIELDS:
        df[field] = df[field].apply(clean_excel_quote)
    print_success("Address fields cleaned")

    total = len(df)
    if LIMIT > 0:
        total = min(LIMIT, total)
        print_warning(f"Processing limited to {LIMIT} rows (out of {len(df)} total)")
    else:
        print(f"Processing {total} addresses")

    # Load cache
    cache = load_cache()
    cache_lock = threading.Lock()

    # ========================================================================
    # PARALLEL GEOCODING
    # ========================================================================

    print_header("Geocoding Addresses (Parallel)")
    
    x_3879 = [None] * total
    y_3879 = [None] * total
    errors = []

    # Build parameters template (avoid recreating in loop)
    params_template = {
        "size": 10,
        "lang": "fi",
        "digitransit-subscription-key": API_KEY,
        "sources": "nlsfi,oa,osm",
        "boundary.rect.min_lon": 24.497,
        "boundary.rect.max_lon": 24.943,
        "boundary.rect.min_lat": 59.899,
        "boundary.rect.max_lat": 60.365
    }

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks
        futures = {}
        for idx in range(total):
            row = df.iloc[idx]
            future = executor.submit(
                geocode_address_worker,
                idx, row, ADDRESS_FIELDS, params_template, transformer, REQUEST_DELAY, cache, cache_lock
            )
            futures[future] = idx
        
        # Collect results with progress bar
        with tqdm(total=total, unit="address", desc="Geocoding", ncols=80) as pbar:
            for future in as_completed(futures):
                try:
                    idx, lon, lat, error_msg = future.result()
                    
                    x_3879[idx] = lon
                    y_3879[idx] = lat
                    
                    if error_msg:
                        address = df.iloc[idx].get(ADDRESS_FIELDS[0], "")
                        errors.append((idx + 1, str(address)[:50], error_msg))
                    
                    pbar.update(1)
                except Exception as e:
                    print_error(f"Worker error: {e}")
                    pbar.update(1)

    # Save cache
    save_cache(cache)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print_header("Saving Results")
    
    df_output = df.copy()
    df_output['x_3879'] = None
    df_output['y_3879'] = None
    df_output.loc[:total-1, 'x_3879'] = x_3879
    df_output.loc[:total-1, 'y_3879'] = y_3879

    output_csv = input_path.parent / "Geocode" / f"geocode_{input_path.name}"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    df_output.to_csv(output_csv, index=False, sep=",", encoding=used_encoding)
    print_success(f"Results saved to: {output_csv}")

    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================

    elapsed = time.time() - start_time
    successfully_geocoded = total - len(errors)
    success_rate = (successfully_geocoded / total * 100) if total > 0 else 0

    print_header("Summary Report")
    print(f"Total addresses processed: {total}")
    print(f"Successfully geocoded: {successfully_geocoded} ({success_rate:.1f}%)")
    print(f"Errors: {len(errors)} ({100 - success_rate:.1f}%)")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    if total > 0:
        print(f"Average time per address: {elapsed/total*1000:.1f}ms")

    if errors:
        print(f"\nFirst 10 errors:")
        for row_num, address, error in errors[:10]:
            print(f"  Row {row_num}: {error}")
            print(f"           Address: {address}...")
        
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print_success("No errors during geocoding!")

    print(f"\nFile encoding: {used_encoding}")
    print(f"Delimiter: '{detected_delim}'")
    print(f"Cache entries: {len(cache)}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()