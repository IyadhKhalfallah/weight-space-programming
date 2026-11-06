"""E7 data: real hourly weather (Open-Meteo archive) for 10 European cities, 2024-01-01 .. 2026-10-31.
One row per city-day D: features describe day D; label = rain on day D+1 (>= 1 mm).
Train period: rows before 2026-07-01. Test period: 2026-07-01 .. 2026-10-30 (after the model-writer's knowledge cutoff).
"""
import json, urllib.request, csv, os
import numpy as np
CITIES = {"Paris": (48.85, 2.35), "London": (51.51, -0.13), "Amsterdam": (52.37, 4.90), "Berlin": (52.52, 13.40),
          "Brussels": (50.85, 4.35), "Dublin": (53.35, -6.26), "Madrid": (40.42, -3.70), "Rome": (41.90, 12.50),
          "Hamburg": (53.55, 9.99), "Lyon": (45.76, 4.84)}
VARS = "pressure_msl,relative_humidity_2m,cloud_cover,wind_speed_10m,wind_direction_10m,temperature_2m,dew_point_2m,precipitation"
os.makedirs("experiments/e7", exist_ok=True); rows = []
for city, (lat, lon) in CITIES.items():
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2024-01-01"
           f"&end_date=2026-10-31&hourly={VARS}&timezone=UTC")
    h = json.load(urllib.request.urlopen(url, timeout=120))["hourly"]
    a = {k: np.array([np.nan if v is None else v for v in h[k]], float) for k in VARS.split(",")}
    days = len(h["time"]) // 24; g = lambda k: a[k][:days * 24].reshape(days, 24)
    P, RH, CC, WS, WD, T, TD, PR = (g(k) for k in VARS.split(","))
    rain = np.nansum(PR, 1)
    for d in range(1, days - 1):
        date = h["time"][d * 24][:10]
        wdir = np.deg2rad(np.nanmean(WD[d, 18:])); ws = np.nanmean(WS[d, 18:])
        rows.append({"city": city, "date": date, "month": int(date[5:7]), "lat": lat,
                     "pressure": np.nanmean(P[d]), "pressure_change_24h": P[d, 23] - P[d - 1, 23],
                     "pressure_change_6h": P[d, 23] - P[d, 17], "humidity": np.nanmean(RH[d]), "humidity_evening": np.nanmean(RH[d, 18:]),
                     "cloud": np.nanmean(CC[d]), "cloud_evening": np.nanmean(CC[d, 18:]), "wind_speed_evening": ws,
                     "wind_from_west": -np.sin(wdir), "wind_from_south": -np.cos(wdir),   # +1 = wind blowing FROM west / south
                     "temp": np.nanmean(T[d]), "dewpoint_spread_evening": np.nanmean(T[d, 18:] - TD[d, 18:]),
                     "rain_today": rain[d], "rain_evening": np.nansum(PR[d, 18:]),
                     "rain_tomorrow": int(rain[d + 1] >= 1.0)})
    print(f"{city}: {days} days")
keys = list(rows[0])
with open("experiments/e7/weather.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
print(len(rows), "rows; test rows:", sum(r["date"] >= "2026-07-01" for r in rows))
