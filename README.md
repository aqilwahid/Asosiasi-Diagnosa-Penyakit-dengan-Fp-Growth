# Asosiasi Diagnosa Penyakit dengan FP-Growth

Proyek ini menggunakan algoritma **FP-Growth** untuk menemukan asosiasi antara berbagai gejala dan diagnosa penyakit dalam dataset medis. FP-Growth adalah salah satu algoritma yang paling efisien untuk **frequent pattern mining**, yang digunakan untuk menemukan pola asosiasi antara item-item dalam dataset, seperti hubungan antara gejala pasien dan penyakit yang diderita.

## Latar Belakang

Dalam dunia medis, menemukan pola asosiasi antara gejala dan diagnosa penyakit bisa menjadi alat yang sangat berguna. Dengan menggunakan algoritma seperti FP-Growth, kita dapat menemukan pola atau hubungan tersembunyi yang mungkin tidak terdeteksi secara manual. Proyek ini bertujuan untuk membantu dalam pembuatan **sistem pendukung keputusan medis** yang dapat memberikan rekomendasi penyakit berdasarkan gejala yang dihadapi pasien.

## Algoritma FP-Growth

**FP-Growth** adalah algoritma yang sangat efisien untuk menemukan **frequent itemsets** tanpa perlu melakukan banyak iterasi seperti algoritma Apriori. FP-Growth bekerja dengan cara:

1. Membuat **Frequent Pattern Tree (FP-Tree)** dari dataset.
2. Menggunakan FP-Tree untuk mengekstrak frequent itemsets.

Algoritma ini mengurangi jumlah iterasi pada dataset dan memori yang digunakan, sehingga sangat cocok untuk dataset besar.

## Struktur Proyek

```
|-- Index.py
|-- README.md
|-- requirements.txt
```

## Dataset

Dataset yang digunakan dalam proyek ini berisi informasi medis, seperti:
- **ID Pasien**
- **Gejala**: Sekumpulan gejala yang dialami pasien.
- **Diagnosa**: Penyakit atau kondisi medis yang didiagnosis berdasarkan gejala tersebut.

Dataset disimpan dalam format CSV dan memerlukan preprocessing sebelum diterapkan FP-Growth.

## Instalasi

1. Clone repositori ini:

   ```bash
   git clone https://github.com/aqilwahid/Asosiasi-Diagnosa-Penyakit-dengan-Fp-Growth.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan Jupyter Notebook untuk mengeksplorasi dataset dan menjalankan FP-Growth:

   ```bash
   jupyter notebook notebooks/FP_Growth_Medical.ipynb
   ```

## Cara Penggunaan

1. **Preprocessing Data**: Script `preprocessing.py` digunakan untuk membersihkan dan menyiapkan dataset medis. Data perlu diubah menjadi format yang dapat diterima oleh algoritma FP-Growth.
   
2. **Penerapan FP-Growth**: Script `fp_growth.py` berisi implementasi FP-Growth yang menggunakan pustaka `mlxtend` untuk mengekstrak frequent itemsets dari dataset gejala dan diagnosa.

3. **Mengekstrak Aturan Asosiasi**: Setelah mendapatkan frequent itemsets, script `association_rules.py` digunakan untuk mengekstrak aturan asosiasi yang dapat membantu dalam mendiagnosa penyakit berdasarkan gejala.

4. **Visualisasi dan Interpretasi**: Jupyter notebook berisi proses visualisasi dari frequent itemsets dan aturan asosiasi yang ditemukan.

## Hasil

Setelah menjalankan algoritma FP-Growth, kita dapat menemukan pola asosiasi antara gejala-gejala tertentu dan diagnosa penyakit. Pola-pola ini dapat digunakan untuk membantu dalam pengambilan keputusan medis, memberikan rekomendasi diagnosa, atau mengembangkan sistem diagnosis berbasis gejala.

## Dependencies

Proyek ini memerlukan beberapa pustaka Python yang bisa diinstall melalui `requirements.txt`:

- `pandas`
- `mlxtend`
- `matplotlib`
- `numpy`
