"""
Studi kasus IoT Cold Chain - PT Maju Bersama.
Mengikuti alur materi: CRISP-DM + 4 level analitik.

Jalankan:  python src/analysis.py
Output  :  outputs/*.png dan outputs/report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
log = []


def say(text=""):
    print(text)
    log.append(text)


# ---------------------------------------------------------------- 1. BUSINESS UNDERSTANDING
say("# Laporan Studi Kasus: IoT Cold Chain PT Maju Bersama\n")
say("## 1. Business understanding")
say("Pertanyaan bisnis (pimpinan): *\"Apakah penyimpanan dingin kita aman dan bisa dicegah dari kerugian?\"*")
say("Diterjemahkan menjadi pertanyaan analitik:")
say("- Deskriptif  : seberapa sering suhu melewati batas, di sensor/jam mana?")
say("- Diagnostik  : apa penyebab excursion dan penyebab mana yang paling merugikan?")
say("- Prediktif   : bisakah breach 30 menit ke depan diprediksi?")
say("- Preskriptif : pada ambang alarm berapa biaya total paling rendah?\n")

# ---------------------------------------------------------------- 2. DATA UNDERSTANDING
say("## 2. Data understanding")
sensors = pd.read_csv(DATA / "sensors.csv")
readings = pd.read_csv(DATA / "readings.csv", parse_dates=["ts"])
incidents = pd.read_csv(DATA / "incidents.csv", parse_dates=["start_ts"])
say(f"Granularitas: sensors={len(sensors)} baris (per sensor), readings={len(readings)} baris "
    f"(per sensor per 10 menit), incidents={len(incidents)} baris (per kejadian).")
say(f"Kualitas data mentah: missing temp={readings.temp_c.isna().sum()}, "
    f"duplikat={readings.duplicated(['sensor_id', 'ts']).sum()}, "
    f"glitch (<-50 atau >60 C)={((readings.temp_c < -50) | (readings.temp_c > 60)).sum()}")

# Jebakan join (fan-out): menggabung tabel dengan granularitas berbeda
bad = readings.merge(incidents, on="sensor_id", how="left")
say(f"\n**Jebakan join**: readings JOIN incidents hanya lewat sensor_id -> {len(readings)} baris "
    f"membengkak jadi {len(bad)} baris (fan-out x{len(bad) / len(readings):.1f}). "
    "Rata-rata/jumlah apa pun setelah join ini akan salah. Tabel incidents TIDAK di-join mentah ke readings.\n")
del bad

# ---------------------------------------------------------------- 3. DATA PREPARATION
say("## 3. Data preparation (garbage in, garbage out)")
n0 = len(readings)
df = readings.drop_duplicates(["sensor_id", "ts"]).copy()
say(f"- Hapus duplikat: {n0} -> {len(df)} baris")
df.loc[(df.temp_c < -50) | (df.temp_c > 60), "temp_c"] = np.nan   # glitch sensor
df = df.sort_values(["sensor_id", "ts"]).reset_index(drop=True)
n_na = int(df.temp_c.isna().sum())
df["temp_c"] = df.groupby("sensor_id")["temp_c"].transform(
    lambda s: s.interpolate(limit=3, limit_direction="both"))
say(f"- Glitch -> NaN lalu interpolasi per sensor (maks 3 titik): NaN {n_na} -> {int(df.temp_c.isna().sum())}")
df = df.dropna(subset=["temp_c"])
df = df.merge(sensors, on="sensor_id", how="left", validate="many_to_one")  # join aman: many-to-one
assert len(df) == len(df.drop_duplicates(["sensor_id", "ts"]))
say(f"- Join sensors (many-to-one, divalidasi): {len(df)} baris, tidak ada fan-out")
df["breach"] = (df.temp_c > df.limit_c).astype(int)
say(f"- Label breach: {df.breach.mean():.2%} baris -> **data tidak seimbang** "
    "(bukti sedikit), akurasi biasa menyesatkan; pakai precision/recall.\n")

# ---------------------------------------------------------------- 4. DESKRIPTIF
say("## 4. Analitik deskriptif: apa yang terjadi?")
by_sensor = df.groupby(["sensor_id", "zone"]).agg(
    mean_temp=("temp_c", "mean"), breach_rate=("breach", "mean")).round(4)
say(by_sensor.to_markdown())
by_hour = df.groupby(df.ts.dt.hour).breach.mean()
say(f"\nJam dengan breach tertinggi: {by_hour.idxmax():02d}:00 ({by_hour.max():.2%}).\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
daily = df.groupby([df.ts.dt.date, "zone"]).breach.mean().unstack()
daily.plot(ax=ax[0], marker="o", ms=3)
ax[0].set_title("Persentase breach harian"); ax[0].set_xlabel(""); ax[0].set_ylabel("rasio breach")
by_hour.plot(kind="bar", ax=ax[1], color="#4c78a8")
ax[1].set_title("Breach per jam"); ax[1].set_xlabel("jam")
plt.tight_layout(); plt.savefig(OUT / "01_deskriptif.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 5. DIAGNOSTIK
say("## 5. Analitik diagnostik: mengapa terjadi?")
by_cause = incidents.groupby("cause").agg(
    kejadian=("incident_id", "count"),
    total_kerugian_juta=("loss_idr", lambda x: round(x.sum() / 1e6, 1)),
    rata2_durasi_menit=("duration_min", "mean")).round(1).sort_values("total_kerugian_juta", ascending=False)
say(by_cause.to_markdown())
top = by_cause.index[0]
say(f"\nPenyebab paling merugikan: **{top}** "
    f"({by_cause.total_kerugian_juta.iloc[0] / by_cause.total_kerugian_juta.sum():.0%} dari total kerugian).")
say(f"Rasio breach saat pintu terbuka = {df[df.door_open == 1].breach.mean():.2%} "
    f"vs pintu tertutup = {df[df.door_open == 0].breach.mean():.2%}.\n")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
by_cause.total_kerugian_juta.plot(kind="barh", ax=ax, color="#e45756")
ax.set_xlabel("Total kerugian (juta Rp)"); ax.set_ylabel(""); ax.invert_yaxis()
ax.set_title("Kerugian per penyebab")
plt.tight_layout(); plt.savefig(OUT / "02_diagnostik.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 6. PREDIKTIF
say("## 6. Analitik prediktif: apa yang akan terjadi?")
say("Target: suhu melewati batas dalam 30 menit ke depan, hanya pada baris yang SAAT INI masih aman "
    "(kalau tidak, modelnya cuma membaca suhu yang sudah breach).")
g = df.groupby("sensor_id")
df["margin"] = df.limit_c - df.temp_c
df["slope3"] = (df.temp_c - g.temp_c.shift(3)) / 3
df["roll_mean6"] = g.temp_c.transform(lambda s: s.rolling(6).mean())
df["roll_max6"] = g.temp_c.transform(lambda s: s.rolling(6).max())
df["door_last6"] = g.door_open.transform(lambda s: s.rolling(6).sum())
df["hour"] = df.ts.dt.hour
df["is_frozen"] = (df.zone == "Frozen").astype(int)
future = pd.concat([g.breach.shift(-k) for k in (1, 2, 3)], axis=1)
df["target"] = future.max(axis=1)
FEATS = ["margin", "slope3", "roll_mean6", "roll_max6", "door_open", "door_last6",
         "humidity_pct", "hour", "is_frozen"]
model_df = df[(df.breach == 0)].dropna(subset=FEATS + ["target"]).copy()
cut = model_df.ts.min() + pd.Timedelta(days=21)          # split berdasarkan WAKTU, bukan acak
train, test = model_df[model_df.ts < cut], model_df[model_df.ts >= cut]
say(f"Split waktu: train={len(train)} (21 hari pertama), test={len(test)} (9 hari terakhir); "
    f"proporsi positif test={test.target.mean():.2%}")

rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=42)
rf.fit(train[FEATS], train.target)
proba = rf.predict_proba(test[FEATS])[:, 1]
pred = (proba >= 0.5).astype(int)
base = (test.margin < 2.0).astype(int)                    # baseline sederhana
say("\n| Model | Precision | Recall | PR-AUC |\n|---|---|---|---|")
say(f"| Baseline (margin < 2C) | {precision_score(test.target, base, zero_division=0):.2f} | "
    f"{recall_score(test.target, base):.2f} | - |")
say(f"| RandomForest (thr 0.5) | {precision_score(test.target, pred, zero_division=0):.2f} | "
    f"{recall_score(test.target, pred):.2f} | {average_precision_score(test.target, proba):.2f} |")
imp = pd.Series(rf.feature_importances_, FEATS).sort_values(ascending=False)
say("\nFitur terpenting: " + ", ".join(f"{k} ({v:.2f})" for k, v in imp.head(4).items()))
say("\n> Evaluasi: hasil model perlu diverifikasi manusia. Data sintetik membuat angka lebih mudah "
    "dari dunia nyata; jangan klaim performa ini untuk data sungguhan.\n")

# ---------------------------------------------------------------- 7. PRESKRIPTIF
say("## 7. Analitik preskriptif: apa yang harus dilakukan?")
COST_FP, COST_FN = 100_000, 500_000    # ASUMSI, ubah sesuai kondisi nyata (Rp)
say(f"Asumsi biaya: alarm palsu (cek teknisi) = Rp{COST_FP:,}; breach terlewat = Rp{COST_FN:,}.")
thr = np.linspace(0.05, 0.95, 91)
costs = []
for t in thr:
    p = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(test.target, p, labels=[0, 1]).ravel()
    costs.append(fp * COST_FP + fn * COST_FN)
costs = np.array(costs)
best = thr[costs.argmin()]
cost_default = costs[np.argmin(abs(thr - 0.5))]
say(f"Ambang alarm optimal = **{best:.2f}** dengan biaya Rp{costs.min():,.0f} "
    f"(vs Rp{cost_default:,.0f} pada ambang 0.5; hemat {1 - costs.min() / cost_default:.0%}).")
say("\nRekomendasi untuk manajemen:")
say(f"1. Aktifkan alarm dini pada probabilitas >= {best:.2f}.")
say(f"2. Prioritaskan pencegahan penyebab '{top}' (kerugian terbesar).")
say("3. Pantau: bandingkan jumlah breach sebelum dan sesudah alarm dipasang, lalu iterasi (CRISP-DM).\n")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(thr, costs / 1e6)
ax.axvline(best, color="red", ls="--", label=f"optimal = {best:.2f}")
ax.set_xlabel("ambang probabilitas alarm"); ax.set_ylabel("biaya total (juta Rp)")
ax.set_title("Simulasi biaya vs ambang alarm"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "03_preskriptif.png", dpi=130); plt.close()

(OUT / "report.md").write_text("\n".join(log), encoding="utf-8")
print(f"\nSelesai. Laporan: {OUT / 'report.md'}")
