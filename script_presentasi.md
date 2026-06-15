# Script Presentasi Proyek Machine Learning

## Pembuka

Assalamu'alaikum warahmatullahi wabarakatuh.

Pada proyek ini, kami membuat model machine learning untuk mengklasifikasikan spesies bunga Iris. Dataset yang digunakan adalah Iris Dataset dari scikit-learn. Model memprediksi spesies bunga berdasarkan empat fitur, yaitu sepal length, sepal width, petal length, dan petal width.

## Business Understanding

Masalah yang diselesaikan adalah bagaimana model dapat mengelompokkan bunga Iris ke dalam tiga spesies, yaitu setosa, versicolor, dan virginica. Tujuan proyek ini adalah membuat model klasifikasi, membandingkan Decision Tree dan Random Forest, serta membuat aplikasi deployment menggunakan Streamlit.

## Data Understanding

Dataset Iris memiliki 150 data, 4 fitur, dan 3 kelas target. Setiap kelas memiliki 50 data, sehingga dataset ini seimbang. Tidak ada missing value dan seluruh fitur berbentuk numerik.

## Data Preparation

Tahap data preparation dilakukan dengan memisahkan fitur dan target. Data dibagi menjadi 80 persen data latih dan 20 persen data uji. Pembagian dilakukan dengan stratify agar distribusi kelas tetap seimbang.

## Modeling

Model pertama adalah Decision Tree. Model ini mudah dijelaskan karena berbentuk pohon keputusan.

Model kedua adalah Random Forest. Model ini menggunakan banyak Decision Tree, sehingga hasil prediksi biasanya lebih stabil.

## Evaluation

Hasil evaluasi menunjukkan bahwa Decision Tree dan Random Forest memperoleh performa yang baik. Kedua model memperoleh accuracy sebesar 0.966667. Nilai precision, recall, dan F1-score juga menunjukkan hasil yang seimbang.

## Deployment

Deployment dibuat menggunakan Streamlit. Pengguna dapat memasukkan nilai sepal length, sepal width, petal length, dan petal width. Setelah memilih model dan menekan tombol prediksi, aplikasi menampilkan spesies Iris yang diprediksi.

## Penutup

Kesimpulannya, proyek ini berhasil membangun model klasifikasi Iris dan menampilkan model dalam bentuk aplikasi sederhana berbasis Streamlit.

Wassalamu'alaikum warahmatullahi wabarakatuh.
