import os
import time
import requests
import pandas as pd
from pathlib import Path
from pyproj import Transformer

"""
Geocode addresses using Digitransit API. Coordinates are output in GK25FIN (EPSG:3879).

Steps:
1. Set INPUT_CSV to your source CSV file path
2. Set ADDRESS_FIELDS = ["Street", "City", ...]
3. Add your Digitransit API_KEY
4. If needed, change separator in pd.read_csv and to_csv 
5. For a test run, set LIMIT > 0, change back to "0" to process all rows

Output:
- Output is saved as 'Geocode/geocode_<inputfilename>.csv' in the same folder as input CSV.
- Adds columns: 'x_3879', 'y_3879' to the end of the CSV

Dependencies: pandas, requests, pyproj
"""

INPUT_CSV = r"INPUT_CSV_PATH_HERE"

ADDRESS_FIELDS = ["s2k3: Yksikön osoite:", "s2k2: Yksikön nimi:"]

API_KEY = "8b2808e2bbd64c7694b488d536f70bc5"

# For testing: Limit the number of addresses to geocode (from the start of the CSV). Use 0 to process all rows
LIMIT = 0

# Coordinate transformer: WGS84 -> EPSG:3879
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3879", always_xy=True)

# Read input
df = pd.read_csv(INPUT_CSV, sep=',', encoding='utf-8-sig')

def clean_excel_quote(s):
    s = str(s or '').strip()
    if s.startswith('="') and s.endswith('"'):
        return s[2:-1]
    return s

# Clean all address fields in the DataFrame
for field in ADDRESS_FIELDS:
    df[field] = df[field].apply(clean_excel_quote)


total = len(df)
x_3879 = []
y_3879 = []

errors = []

for idx, row in df.iterrows():
    if LIMIT != 0 and idx >= LIMIT: # Check if LIMIT is set and reached
        break
    
    address = " ".join(str(row.get(field, '') or '') for field in ADDRESS_FIELDS).strip()

    # Digitransit API query parameters: (https://digitransit.fi/en/developers/apis/3-geocoding-api/address-search/)
    params = {
        "text": address,
        "size": 10,
        "lang": "fi",
        "digitransit-subscription-key": API_KEY,  # key as URL param
        "sources": "nlsfi,oa,osm",
        # specify that the search should only cover the area of the bounding box around the area of Espoo:
        "boundary.rect.min_lon": 24.497,
        "boundary.rect.max_lon": 24.943,
        "boundary.rect.min_lat": 59.899,
        "boundary.rect.max_lat": 60.365
    }

    try:
        response = requests.get(
            "https://api.digitransit.fi/geocoding/v1/search",
            params = params,
            timeout = 10  # <-- Timeout added here (in seconds)
        )
        response.raise_for_status()  # raises HTTPError for bad responses
        data = response.json()

        features = data.get('features', [])
        if features:
            coords = features[0]['geometry']['coordinates']
            lon, lat = coords
            lon_3879, lat_3879 = transformer.transform(lon, lat)
        else:
            lon_3879 = lat_3879 = None
            msg = f"No match found for: '{address}'"
            errors.append(msg)
            print(msg)

    except requests.exceptions.RequestException as e:
        lon_3879 = lat_3879 = None
        msg = f"Network/API error for address '{address}': {e}"
        errors.append(msg)
        print(msg)
    except ValueError as e:
        lon_3879 = lat_3879 = None
        msg = f"Error decoding JSON for address '{address}': {e}"
        errors.append(msg)
        print(msg)
    except Exception as e:
        lon_3879 = lat_3879 = None
        msg = f"Unexpected error for address '{address}': {e}"
        errors.append(msg)
        print(msg)

    x_3879.append(lon_3879)
    y_3879.append(lat_3879)

    print(f"{idx + 1}/{total} geocoded ({(idx + 1) / total:.1%})")
    time.sleep(0.1)

# For rows not geocoded (beyond LIMIT), append None to keep lengths equal
if LIMIT != 0:
    for _ in range(LIMIT, total):
        x_3879.append(None)
        y_3879.append(None)

# Append new columns to original DataFrame
df['x_3879'] = x_3879
df['y_3879'] = y_3879

# Save results
input_path = Path(INPUT_CSV)
output_csv = input_path.parent / "Geocode" / f"geocode_{input_path.name}"
output_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False,  sep=',', encoding='utf-8-sig')

if errors:
    print(f"Errors during geocoding: {', '.join(errors)}")
else:
    print("No errors during geocoding.")

print(f"Geocoding complete. Results saved to '{output_csv}'.")
