# Laporan Studi Kasus: IoT Cold Chain PT Maju Bersama

## 1. Business understanding
Pertanyaan bisnis (pimpinan): *"Apakah penyimpanan dingin kita aman dan bisa dicegah dari kerugian?"*
Diterjemahkan menjadi pertanyaan analitik:
- Deskriptif  : seberapa sering suhu melewati batas, di sensor/jam mana?
- Diagnostik  : apa penyebab excursion dan penyebab mana yang paling merugikan?
- Prediktif   : bisakah breach 30 menit ke depan diprediksi?
- Preskriptif : pada ambang alarm berapa biaya total paling rendah?

## 2. Data understanding
Granularitas: sensors=8 baris (per sensor), readings=34732 baris (per sensor per 10 menit), incidents=116 baris (per kejadian).
Kualitas data mentah: missing temp=343, duplikat=172, glitch (<-50 atau >60 C)=104

**Jebakan join**: readings JOIN incidents hanya lewat sensor_id -> 34732 baris membengkak jadi 503630 baris (fan-out x14.5). Rata-rata/jumlah apa pun setelah join ini akan salah. Tabel incidents TIDAK di-join mentah ke readings.

## 3. Data preparation (garbage in, garbage out)
- Hapus duplikat: 34732 -> 34560 baris
- Glitch -> NaN lalu interpolasi per sensor (maks 3 titik): NaN 446 -> 0
- Join sensors (many-to-one, divalidasi): 34560 baris, tidak ada fan-out
- Label breach: 3.11% baris -> **data tidak seimbang** (bukti sedikit), akurasi biasa menyesatkan; pakai precision/recall.

## 4. Analitik deskriptif: apa yang terjadi?
|                    |   mean_temp |   breach_rate |
|:-------------------|------------:|--------------:|
| ('S01', 'Chiller') |      4.3395 |        0.0315 |
| ('S02', 'Chiller') |      4.33   |        0.0292 |
| ('S03', 'Chiller') |      4.3264 |        0.0319 |
| ('S04', 'Chiller') |      4.3822 |        0.037  |
| ('S05', 'Frozen')  |    -19.7095 |        0.0218 |
| ('S06', 'Frozen')  |    -19.5937 |        0.0345 |
| ('S07', 'Frozen')  |    -19.6402 |        0.031  |
| ('S08', 'Frozen')  |    -19.6313 |        0.0317 |

Jam dengan breach tertinggi: 16:00 (5.21%).

## 5. Analitik diagnostik: mengapa terjadi?
| cause                |   kejadian |   total_kerugian_juta |   rata2_durasi_menit |
|:---------------------|-----------:|----------------------:|---------------------:|
| Pintu terbuka lama   |         56 |                 632.9 |                110.5 |
| Kompresor bermasalah |         43 |                 545.7 |                121.2 |
| Listrik padam        |         17 |                 218.7 |                114.1 |

Penyebab paling merugikan: **Pintu terbuka lama** (45% dari total kerugian).
Rasio breach saat pintu terbuka = 15.42% vs pintu tertutup = 1.77%.

## 6. Analitik prediktif: apa yang akan terjadi?
Target: suhu melewati batas dalam 30 menit ke depan, hanya pada baris yang SAAT INI masih aman (kalau tidak, modelnya cuma membaca suhu yang sudah breach).
Split waktu: train=23431 (21 hari pertama), test=10007 (9 hari terakhir); proporsi positif test=0.96%

| Model | Precision | Recall | PR-AUC |
|---|---|---|---|
| Baseline (margin < 2C) | 0.42 | 0.32 | - |
| RandomForest (thr 0.5) | 0.91 | 0.52 | 0.64 |

Fitur terpenting: slope3 (0.30), margin (0.21), roll_max6 (0.13), roll_mean6 (0.10)

> Evaluasi: hasil model perlu diverifikasi manusia. Data sintetik membuat angka lebih mudah dari dunia nyata; jangan klaim performa ini untuk data sungguhan.

## 7. Analitik preskriptif: apa yang harus dilakukan?
Asumsi biaya: alarm palsu (cek teknisi) = Rp100,000; breach terlewat = Rp500,000.
Ambang alarm optimal = **0.27** dengan biaya Rp21,000,000 (vs Rp23,500,000 pada ambang 0.5; hemat 11%).

Rekomendasi untuk manajemen:
1. Aktifkan alarm dini pada probabilitas >= 0.27.
2. Prioritaskan pencegahan penyebab 'Pintu terbuka lama' (kerugian terbesar).
3. Pantau: bandingkan jumlah breach sebelum dan sesudah alarm dipasang, lalu iterasi (CRISP-DM).
