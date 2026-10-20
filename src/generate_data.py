"""
Membuat dataset IoT sintetik untuk studi kasus PT Maju Bersama (cold chain).

Tiga tabel dengan granularitas BERBEDA (sengaja, sesuai materi):
  - sensors.csv   : 1 baris per sensor            (8 baris)
  - readings.csv  : 1 baris per sensor per 10 mnt (~34 ribu baris)
  - incidents.csv : 1 baris per kejadian excursion (puluhan baris)

Data sengaja dibuat "berantakan" (missing, duplikat, glitch sensor)
agar tahap data preparation bermakna: garbage in, garbage out.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
DATA = Path(__file__).resolve().parent.parent / "data"

ZONES = {
    "Chiller": {"setpoint": 4.0, "limit": 8.0, "loss_per_hour": 6_000_000},
    "Frozen": {"setpoint": -20.0, "limit": -15.0, "loss_per_hour": 12_000_000},
}
CAUSES = ["Pintu terbuka lama", "Kompresor bermasalah", "Listrik padam"]


def main():
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)

    sensors = pd.DataFrame({
        "sensor_id": [f"S{i:02d}" for i in range(1, 9)],
        "zone": ["Chiller"] * 4 + ["Frozen"] * 4,
        "warehouse": ["Gudang A", "Gudang A", "Gudang B", "Gudang B"] * 2,
    })
    sensors["setpoint_c"] = sensors["zone"].map(lambda z: ZONES[z]["setpoint"])
    sensors["limit_c"] = sensors["zone"].map(lambda z: ZONES[z]["limit"])

    ts = pd.date_range("2026-09-01", periods=30 * 144, freq="10min")  # 30 hari
    rows, incidents = [], []

    for _, s in sensors.iterrows():
        n = len(ts)
        hour = ts.hour.values + ts.minute.values / 60
        busy = ((hour >= 7) & (hour <= 17)).astype(float)   # jam sibuk
        door = (rng.random(n) < (0.03 + 0.12 * busy)).astype(int)

        temp = (s.setpoint_c
                + 0.4 * np.sin(2 * np.pi * hour / 24)
                + rng.normal(0, 0.35, n)
                + 0.8 * door)
        humidity = np.clip(60 + 8 * np.sin(2 * np.pi * (hour - 3) / 24)
                           + rng.normal(0, 2, n) + 3 * door, 30, 95)

        # excursion: pendingin gagal -> suhu naik bertahap selama 1-3 jam
        i = int(rng.integers(20, 80))
        while i < n - 40:
            dur = int(rng.integers(6, 19))                   # 60-180 menit
            cause = rng.choice(CAUSES, p=[0.45, 0.35, 0.20])
            peak = (s.limit_c - s.setpoint_c) + rng.uniform(2.0, 6.0)
            ramp = np.minimum(np.arange(dur) / 4, 1.0) * peak
            if cause == "Pintu terbuka lama":
                door[i:i + dur] = 1
            temp[i:i + dur] += ramp
            temp[i + dur:i + dur + 4] += np.linspace(peak * 0.5, 0, 4)
            breach_len = int((temp[i:i + dur] > s.limit_c).sum())
            if breach_len > 0:
                loss = ZONES[s.zone]["loss_per_hour"] * (breach_len * 10 / 60)
                incidents.append({
                    "incident_id": None, "sensor_id": s.sensor_id,
                    "start_ts": ts[i], "duration_min": dur * 10,
                    "cause": cause, "loss_idr": int(loss * rng.uniform(0.6, 1.2)),
                })
            i += dur + int(rng.integers(150, 420))           # jeda antar kejadian

        rows.append(pd.DataFrame({
            "sensor_id": s.sensor_id, "ts": ts,
            "temp_c": temp.round(2), "humidity_pct": humidity.round(1),
            "door_open": door,
        }))

    readings = pd.concat(rows, ignore_index=True)

    # ---- kotori data (realita: data itu berantakan) ----
    n = len(readings)
    readings.loc[rng.choice(n, int(n * 0.010), replace=False), "temp_c"] = np.nan
    glitch = rng.choice(n, int(n * 0.003), replace=False)
    readings.loc[glitch, "temp_c"] = rng.choice([85.0, -127.0], len(glitch))  # error khas sensor
    dups = readings.sample(int(n * 0.005), random_state=SEED)
    readings = pd.concat([readings, dups], ignore_index=True).sample(frac=1, random_state=SEED)

    inc = pd.DataFrame(incidents).sort_values("start_ts").reset_index(drop=True)
    inc["incident_id"] = [f"INC{i + 1:03d}" for i in range(len(inc))]

    sensors.to_csv(DATA / "sensors.csv", index=False)
    readings.to_csv(DATA / "readings.csv", index=False)
    inc.to_csv(DATA / "incidents.csv", index=False)
    print(f"sensors={len(sensors)}  readings={len(readings)}  incidents={len(inc)}")


if __name__ == "__main__":
    main()
