# digitransit-geocode
Simple python script for geocoding Finnish addresses using the Digitransit API

Geocode addresses using Digitransit API. Coordinates are output in GK25FIN (EPSG:3879) by default.

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
