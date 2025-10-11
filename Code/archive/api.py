import requests
import json

url = "https://re.jrc.ec.europa.eu/api/v5_2/DRcalc"
params = {
    'lat': 52.52,
    'lon': 13.405,
    'raddatabase': 'PVGIS-SARAH2',
    'month': 0,  # 0 = annual
    'outputformat': 'json'
}

response = requests.get(url, params=params)
data = response.json()

# Print entire JSON formatted nicely
print(json.dumps(data, indent=2))

# OR just print optimal values
if 'optimal' in data:
    print(f"Optimal tilt (slope): {data['optimal']['slope']}°")
    print(f"Optimal azimuth (aspect): {data['optimal']['aspect']}°")
else:
    print("No 'optimal' field in response.")
