## Analisis HR Attrition & Sistem Prediksi
Proyek ini adalah solusi untuk menganalisis dan memprediksi risiko keluarnya karyawan (Employee Attrition) menggunakan Machine Learning dan Graph Database. Sistem ini membantu departemen HR dalam mengidentifikasi talenta yang berisiko tinggi untuk keluar dan memahami pola hubungan antar karyawan dalam organisasi.

Live Demo: [Analysis Attrition](attrition-analysis.streamlit.app)

## Fitur Kunci
- **Prediksi Machine Learning:** Menggunakan algoritma `CatBoost Classifier` yang telah dioptimasi dengan Optuna untuk memprediksi probabilitas attrition karyawan.
- **Rekayasa Fitur:** Menerapkan logika bisnis HR seperti Satisfaction Index, Career Stability, Loyalty Ratio, dan Income per Age untuk meningkatkan akurasi model.
- **Optimatisasi Threshold:** Menggunakan cutoff threshold yang dikalibrasi (0.279) untuk memaksimalkan F1-Score dan menangkap lebih banyak karyawan berisiko (Recall tinggi).
- **Analisis Graph:** Integrasi dengan Neo4j untuk memetakan hubungan antara Karyawan, Departemen, dan Peran Pekerjaan (Job Role) guna melihat pola attrition secara visual.

## Alur Pekerjaan Teknis
- **Data Preprocessing:** Pembersihan data dan Feature Engineering dari dataset HR-Employee-Attrition.csv.
- **Model Training:** Pelatihan model menggunakan CatBoost dengan strategi Stratified K-Fold Cross Validation.
- **Hyperparameter Tuning:** Pencarian parameter terbaik menggunakan Optuna (mendapatkan akurasi validasi ~86.2%).
- **Ekspor ke Graph:** Hasil prediksi diekspor ke format CSV yang kompatibel dengan Neo4j untuk analisis relasional.
- **Deployment:** Model disajikan melalui antarmuka web interaktif menggunakan Streamlit.

## Performa Model
- **Algoritma:** CatBoost Classifier
- **Akurasi CV Terbaik:** 86.2%
- **Skor ROC AUC:** 0.837
- **Top 5 Prediktor:** OverTime, JobRole, StockOptionLevel, Age, TotalSatisfication.

## Panduan Instalasi Lokal

Ikuti langkah-langkah berikut untuk menjalankan proyek pada mesin lokal Anda.

### Prasyarat
* Python 3.8 atau lebih tinggi.
* Git.

## Langkah Instalasi (macOS / Linux)

1.  **Clone Repositori:**
    ```bash
    git clone https://github.com/citylighxts/attrition-analysis.git
    cd attrition-analysis
    ```

2. **Buat Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```

3. **Aktifkan Environment:**
    ```bash
    source venv/bin/activate
    ```

4. **Install Dependensi:**
    ```bash
    pip3 install -r requirements.txt
    ```

5. **Jalankan Aplikasi:**
    ```bash
    streamlit run app.py
    ```

## Langkah Instalasi (Windows)

1. Buka Command Prompt (CMD) atau PowerShell di folder proyek.
2. Buat environment: `python -m venv venv`
3. Aktifkan: `venv\Scripts\activate`
4. Install: `pip install -r requirements.txt`
5. Jalankan: `streamlit run app.py`
