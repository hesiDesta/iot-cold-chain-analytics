# IoT Cold Chain Analytics: Studi Kasus PT Maju Bersama

Studi kasus berbasis materi *"Make Sense of Data with Analysis and AI"* (Informatics Practitioner Lecture Series, BRIN).
Perusahaan distributor PT Maju Bersama memasang sensor IoT (suhu, kelembapan, pintu) di gudang pendingin dan beku.
Pimpinan bertanya: **"Apakah penyimpanan dingin kita aman, dan bisakah kerugian dicegah?"**

> Semua data **sintetik** (dibuat oleh `src/generate_data.py`), bukan data perusahaan sungguhan.

## Pemetaan ke materi

| Konsep di materi | Implementasi di proyek ini |
|---|---|
| Pertanyaan bisnis -> pertanyaan analitik | Bagian 1 di `analysis.py` |
| CRISP-DM (iteratif) | Struktur skrip: understanding -> preparation -> modeling -> evaluation -> rekomendasi |
| Granularitas & fan-out saat join | 3 tabel beda granularitas; demo join yang membengkak 14x, lalu join aman `many_to_one` |
| Garbage in, garbage out | Hapus duplikat, glitch sensor (85 C / -127 C), interpolasi data hilang |
| Data tidak seimbang (< 5%) | Breach hanya ~3%; evaluasi pakai precision/recall/PR-AUC |
| Analitik deskriptif | Breach per sensor, per hari, per jam |
| Analitik diagnostik | Penyebab excursion dan total kerugian |
| Analitik prediktif | RandomForest memprediksi breach 30 menit ke depan (split berdasarkan waktu) |
| Analitik preskriptif | Simulasi biaya untuk memilih ambang alarm optimal |
| GenAI harus diverifikasi | Catatan evaluasi di laporan; asumsi biaya bisa diubah |

## Struktur

```
src/generate_data.py   # membuat data sintetik -> data/
src/analysis.py        # pipeline lengkap -> outputs/
data/                  # sensors.csv, readings.csv, incidents.csv
outputs/               # report.md + 3 grafik
```

## Cara menjalankan

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py
python src/analysis.py
```

## Hasil ringkas (seed 42)

- Breach suhu ~3% dari seluruh pembacaan; penyebab terbesar: **pintu terbuka lama** (~45% kerugian).
- Model prediktif: precision 0.91, recall 0.52 pada ambang 0.5 (baseline sederhana: 0.42 / 0.32).
- Ambang alarm optimal 0.27 menurunkan biaya ~11% dibanding ambang 0.5.

## Keterbatasan

- Data sintetik: pola penyebab dan suhu dibuat oleh generator, sehingga performa model lebih mudah dari dunia nyata.
- Biaya alarm palsu dan breach terlewat (`COST_FP`, `COST_FN`) adalah asumsi; sesuaikan dengan kondisi nyata.
- Deployment ke perangkat IoT sungguhan (MQTT, ESP32, dsb.) belum dicakup; bisa jadi pengembangan lanjutan.
