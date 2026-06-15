# Machine Learning Terapan

## Project Klasifikasi Spesies Bunga Iris

Proyek ini membahas klasifikasi spesies bunga Iris menggunakan algoritma Decision Tree dan Random Forest. Dataset yang digunakan adalah Iris Dataset dari library scikit-learn. Model dibuat untuk memprediksi spesies bunga berdasarkan empat fitur, yaitu sepal length, sepal width, petal length, dan petal width.

## Daftar Isi Repository

| Bagian | Link |
|---|---|
| Code Notebook | [klasifikasi_iris.ipynb](klasifikasi_iris.ipynb) |
| Laporan Proyek | [laporan_submission_iris.md](laporan_submission_iris.md) |
| Aplikasi Deployment | [app.py](app.py) |
| Hasil Evaluasi | [reports/evaluation_results.md](reports/evaluation_results.md) |

## Informasi Proyek

| Keterangan | Isi |
|---|---|
| Judul | Klasifikasi Spesies Bunga Iris Menggunakan Decision Tree dan Random Forest |
| Dataset | Iris Dataset |
| Sumber Dataset | scikit-learn |
| Algoritma | Decision Tree dan Random Forest |
| Jenis Masalah | Klasifikasi |
| Metodologi | CRISP-DM |
| Deployment | Streamlit |

## Cara Menjalankan Aplikasi

Clone repository:

```bash
git clone https://github.com/Hisham190/proyek-ml-klasifikasi-iris.git
```

Masuk ke folder project:

```bash
cd proyek-ml-klasifikasi-iris
```

Install library:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python -m streamlit run app.py
```

Aplikasi akan berjalan pada browser melalui alamat berikut:

```bash
http://localhost:8501
```

## Struktur Repository

```text
proyek-ml-klasifikasi-iris/
│
├── README.md
├── laporan_submission_iris.md
├── klasifikasi_iris.ipynb
├── app.py
├── train_model.py
├── requirements.txt
│
├── data/
│   └── iris_dataset.csv
│
├── models/
│   ├── decision_tree_model.pkl
│   └── random_forest_model.pkl
│
├── reports/
│   └── evaluation_results.md
│
└── images/
    ├── class_distribution.png
    ├── model_comparison.png
    └── confusion_matrix_random_forest.png
```

## Hasil Singkat

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | 0.966667 | 0.969697 | 0.966667 | 0.966583 |
| Random Forest | 0.966667 | 0.969697 | 0.966667 | 0.966583 |
