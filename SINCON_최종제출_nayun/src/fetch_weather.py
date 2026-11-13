"""인천 시간별 날씨를 Open-Meteo Historical Weather API로 받는다 (인증키 불필요).

- 위치: 인천 도심 (위도 37.46, 경도 126.70) 한 지점
- 기간: 2025-01-01 ~ 2026-10-31 (학습 + 예측 기간)
- 항목: 강수량(mm), 기온(°C), 적설(cm)
- 라이선스: CC BY 4.0, 출처 표시 "Weather data by Open-Meteo.com" (SOURCES.md 기록)

결과: data/external/weather_hourly.csv (date, hour, precip_mm, temp_c, snow_cm)
      data/external/weather_extra_hourly.csv (date, hour, humidity, feels_c, wind_kmh, cloud_pct, radiation, wcode)

사용법 (프로젝트 루트에서):
    python src/fetch_weather.py
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "latitude": 37.46, "longitude": 126.70,
    "start_date": "2025-01-01", "end_date": "2026-10-31",
    "hourly": "precipitation,temperature_2m,snowfall",
    "timezone": "Asia/Seoul",
}


EXTRA = {"relative_humidity_2m": "humidity", "apparent_temperature": "feels_c", "wind_speed_10m": "wind_kmh",
         "cloud_cover": "cloud_pct", "shortwave_radiation": "radiation", "weather_code": "wcode"}


def fetch_extra():
    params = {**PARAMS, "hourly": ",".join(EXTRA)}
    with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(params), timeout=60) as r:
        d = json.load(r)
    h = pd.DataFrame(d["hourly"])
    ts = pd.to_datetime(h["time"])
    out = pd.DataFrame({"date": ts.dt.strftime("%Y-%m-%d"), "hour": ts.dt.hour})
    for k, v in EXTRA.items():
        out[v] = h[k]
    path = Path("data/external/weather_extra_hourly.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path} ({len(out)}행), 결측 {int(out.isna().sum().sum())}개, 기간 {out.date.min()} ~ {out.date.max()}")


def main():
    with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(PARAMS), timeout=60) as r:
        d = json.load(r)
    h = pd.DataFrame(d["hourly"])
    ts = pd.to_datetime(h["time"])
    out = pd.DataFrame({
        "date": ts.dt.strftime("%Y-%m-%d"),
        "hour": ts.dt.hour,
        "precip_mm": h["precipitation"],
        "temp_c": h["temperature_2m"],
        "snow_cm": h["snowfall"],
    })
    path = Path("data/external/weather_hourly.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path} ({len(out)}행), 결측 {int(out.isna().sum().sum())}개, "
          f"기간 {out.date.min()} ~ {out.date.max()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extra":
        fetch_extra()
    else:
        main()
