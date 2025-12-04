## 🧠 HR Attrition Analysis & Prediction System
Proyek ini adalah solusi untuk menganalisis dan memprediksi risiko keluarnya karyawan (Employee Attrition) menggunakan Machine Learning dan Graph Database. Sistem ini membantu departemen HR dalam mengidentifikasi talenta yang berisiko tinggi untuk keluar dan memahami pola hubungan antar karyawan dalam organisasi.

🔗 Live Demo: [Analysis Attrition](attrition-analysis.streamlit.app)

## 🚀 Key Features
- **Machine Learning Prediction:** Menggunakan algoritma `CatBoost Classifier` yang telah dioptimasi dengan Optuna untuk memprediksi probabilitas attrition karyawan.
- **Advanced Feature Engineering:** Menerapkan logika bisnis HR seperti Satisfaction Index, Career Stability, Loyalty Ratio, dan Income per Age untuk meningkatkan akurasi model.
- **Threshold Optimization:** Menggunakan cutoff threshold yang dikalibrasi (0.279) untuk memaksimalkan F1-Score dan menangkap lebih banyak karyawan berisiko (Recall tinggi).
- **Knowledge Graph Analysis:** Integrasi dengan Neo4j untuk memetakan hubungan antara Karyawan, Departemen, dan Peran Pekerjaan (Job Role) guna melihat pola attrition secara visual.

## 📊 Model Performance
- **Algorithm:** CatBoost Classifier
- **Best CV Accuracy:** 86.2%
- **ROC AUC Score:** 0.837
- **Top Predictors:** OverTime, JobRole, StockOption Level, Age, Total Satisfication.

## 💻 Local Installation Guide

Follow these steps to run the project on your local machine.

### Prerequisites
* Python 3.8 or higher.
* Git.

### Installation Steps (macOS / Linux)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/citylighxts/attrition-analysis.git
    cd attrition-analysis
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the Environment:**
    ```bash
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

### Installation Steps (Windows)

1.  Open Command Prompt (CMD) or PowerShell in the project folder.
2.  Create environment: `python -m venv venv`
3.  Activate: `venv\Scripts\activate`
4.  Install: `pip install -r requirements.txt`
5.  Run: `streamlit run app.py`
