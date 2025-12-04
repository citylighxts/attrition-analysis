import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from neo4j import GraphDatabase
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="HR Strategic Dashboard", layout="wide", page_icon="🏢")

st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center;}
    .high-risk {color: #ff4b4b; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Koneksi Database")
    db_uri = st.text_input("URI", "bolt://localhost:7687")
    db_user = st.text_input("Username", "neo4j")
    db_pass = st.text_input("Password", "12345678", type="password")
    db_name = st.text_input("Database Name", "fp-rsbp")
    st.divider()
    st.caption("Kelompok 9 - Analisis Attrition")

@st.cache_resource
def get_driver(uri, user, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except:
        return None

def run_cypher(query, params=None):
    driver = get_driver(db_uri, db_user, db_pass)
    if driver:
        try:
            with driver.session(database=db_name) as session:
                result = session.run(query, params)
                return [r.data() for r in result]
        except Exception as e:
            return []
    return []

@st.cache_resource
def load_ml_models():
    model = CatBoostClassifier()
    feature_names = []
    try:
        model.load_model("catboost_optimized.cbm")
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    except:
        pass
    return model, feature_names

model, feature_names = load_ml_models()

st.title("🏢 HR Strategic Intelligence System")
st.markdown("Sistem pendukung keputusan berbasis **Graph Database** & **Machine Learning** untuk retensi karyawan.")

if not get_driver(db_uri, db_user, db_pass):
    st.error("❌ Gagal terhubung ke Database. Cek Sidebar.")
    st.stop()

kpi_data = run_cypher("""
MATCH (e:Employee)
RETURN 
    count(e) as total, 
    sum(case when e.AttritionRisk >= 0.279 then 1 else 0 end) as risk_count,
    avg(e.AttritionRisk) as avg_risk
""")

if kpi_data:
    kpi = kpi_data[0]
    total = kpi['total']
    risk = kpi['risk_count']
    risk_pct = (risk / total * 100) if total > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Karyawan", f"{total:,}")
    col2.metric("Karyawan High Risk", f"{risk}", delta=f"-{risk_pct:.1f}%", delta_color="inverse")
    col3.metric("Rata-rata Risiko Organisasi", f"{kpi['avg_risk']:.2%}")
    col4.metric("Threshold Model", "27.9%")
    st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Dashboard", 
    "🕸️ Struktur & Relasi", 
    "🧠 Faktor Penyebab (AI)", 
    "📝 Rekomendasi Strategis",
    "🔮 Simulasi Prediksi"
])

with tab1:
    st.subheader("Peta Risiko Departemen")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        q_sunburst = """
        MATCH (e:Employee)
        WHERE e.AttritionRisk >= 0.279
        RETURN e.Department as Dept, e.JobRole as Role, count(e) as Jumlah
        """
        df_sun = pd.DataFrame(run_cypher(q_sunburst))
        
        if not df_sun.empty:
            fig_sun = px.sunburst(
                df_sun, 
                path=['Dept', 'Role'], 
                values='Jumlah',
                color='Jumlah',
                color_continuous_scale='Reds',
                title="Hierarki Karyawan Berisiko Tinggi (Klik untuk Drill-down)"
            )
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.info("Data tidak cukup untuk visualisasi hierarki.")

    with c2:
        st.subheader("Top 5 Job Role Kritis")
        q_role = """
        MATCH (e:Employee)-[:HAS_ROLE]->(r:JobRole)
        WHERE e.AttritionRisk >= 0.279
        RETURN r.name as Role, count(e) as Count
        ORDER BY Count DESC LIMIT 5
        """
        df_role = pd.DataFrame(run_cypher(q_role))
        if not df_role.empty:
            st.dataframe(df_role.style.background_gradient(cmap="Reds"), use_container_width=True)

with tab2:
    st.subheader("Analisis Jaringan Karyawan")
    st.markdown("Mengidentifikasi karyawan kunci dalam jaringan.")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        roles = pd.DataFrame(run_cypher("MATCH (r:JobRole) RETURN DISTINCT r.name as n"))['n'].tolist()
        role_filter = st.selectbox("Filter Job Role:", ["Semua"] + roles)
    
    q_graph = """
    MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
    MATCH (e)-[:HAS_ROLE]->(r:JobRole)
    WHERE e.AttritionRisk >= 0.279
    """
    if role_filter != "Semua":
        q_graph += f" AND r.name = '{role_filter}'"
    
    q_graph += " RETURN e.EmployeeID, r.name as Role, d.name as Dept, e.MonthlyIncome as Gaji, e.AttritionRisk as Risk ORDER BY Risk DESC LIMIT 20"
    
    df_graph = pd.DataFrame(run_cypher(q_graph))
    if not df_graph.empty:
        df_graph['Risk'] = df_graph['Risk'].apply(lambda x: f"{x:.1%}")
        df_graph['Gaji'] = df_graph['Gaji'].apply(lambda x: f"${x:,}")
        
        st.dataframe(df_graph, use_container_width=True)
    else:
        st.info("Tidak ada data karyawan berisiko pada filter ini.")

with tab3:
    st.subheader("🧠 Mengapa Karyawan Keluar?")
    st.caption("Analisis 'Explainable AI' dari Model CatBoost untuk mengetahui akar masalah.")
    
    if model:
        feat_imp = model.get_feature_importance()
        df_imp = pd.DataFrame({
            'Fitur': feature_names,
            'Pentingnya': feat_imp
        }).sort_values(by='Pentingnya', ascending=False).head(10)
        
        fig_imp = px.bar(
            df_imp, x='Pentingnya', y='Fitur', orientation='h',
            title="10 Faktor Utama Penyebab Attrition",
            color='Pentingnya', color_continuous_scale='Blues'
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed")) 
        st.plotly_chart(fig_imp, use_container_width=True)
        
        top_factor = df_imp.iloc[0]['Fitur']
        st.info(f"💡 **Insight:** Faktor **{top_factor}** adalah penentu terbesar keputusan karyawan. Manajemen harus memprioritaskan kebijakan terkait hal ini.")
    else:
        st.error("Model tidak ditemukan.")

with tab4:
    st.subheader("📝 Laporan & Rekomendasi Tindakan")
    
    rekomendasi = []
    
    avg_income_risk = run_cypher("MATCH (e:Employee) WHERE e.AttritionRisk >= 0.279 RETURN avg(e.MonthlyIncome) as inc")[0]['inc']
    avg_income_safe = run_cypher("MATCH (e:Employee) WHERE e.AttritionRisk < 0.279 RETURN avg(e.MonthlyIncome) as inc")[0]['inc']
    
    if avg_income_risk and avg_income_safe:
        gap = (avg_income_safe - avg_income_risk) / avg_income_safe
        if gap > 0.15: 
            rekomendasi.append({
                "Area": "💰 Kompensasi",
                "Status": "KRITIS",
                "Temuan": f"Karyawan berisiko digaji {gap:.1%} lebih rendah dari rata-rata.",
                "Saran": "Lakukan penyesuaian gaji atau berikan bonus retensi untuk Top Talent."
            })
            
    ot_risk = run_cypher("MATCH (e:Employee) WHERE e.AttritionRisk >= 0.279 AND e.OverTime = 'Yes' RETURN count(e) as c")[0]['c']
    if ot_risk > (risk / 2): 
        rekomendasi.append({
            "Area": "⏰ Work-Life Balance",
            "Status": "WARNING",
            "Temuan": "Mayoritas karyawan berisiko sering melakukan lembur (OverTime).",
            "Saran": "Evaluasi beban kerja dan pertimbangkan penambahan headcount atau efisiensi proses."
        })
        
    sat_risk = run_cypher("MATCH (e:Employee) WHERE e.AttritionRisk >= 0.279 RETURN avg(e.EnvironmentSatisfaction) as s")[0]['s']
    if sat_risk is not None and sat_risk < 2.5:
        rekomendasi.append({
            "Area": "🏢 Lingkungan Kerja",
            "Status": "PERLU PERBAIKAN",
            "Temuan": "Skor kepuasan lingkungan rendah pada kelompok berisiko.",
            "Saran": "Lakukan survei internal mendalam atau team bonding."
        })

    if rekomendasi:
        for rec in rekomendasi:
            with st.expander(f"{rec['Area']} - {rec['Status']}", expanded=True):
                st.write(f"**Temuan:** {rec['Temuan']}")
                st.success(f"**Rekomendasi:** {rec['Saran']}")
    else:
        st.success("Berdasarkan data saat ini, tidak ada anomali ekstrem yang terdeteksi secara otomatis.")

    st.download_button("📥 Unduh Laporan PDF", "Laporan Simulasi...", file_name="HR_Attrition_Report.txt")

with tab5:
    st.subheader("🔮 Simulasi Prediksi Karyawan")
    st.caption("Masukkan data profil karyawan untuk memprediksi risiko attrition menggunakan model CatBoost.")
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("### 👤 Profil Diri")
            age = st.number_input("Umur", 18, 60, 30, help="Usia karyawan saat ini")
            marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])
            education = st.selectbox("Tingkat Pendidikan (1-5)", [1, 2, 3, 4, 5])
            edu_field = st.selectbox("Bidang Pendidikan", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
            distance = st.number_input("Jarak ke Kantor (Km)", 1, 50, 10)

        with c2:
            st.markdown("### 💼 Pekerjaan")
            dept = st.selectbox("Departemen", ["Sales", "Research & Development", "Human Resources"])
            role = st.selectbox("Jabatan (Job Role)", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
            job_level = st.selectbox("Job Level (1-5)", [1, 2, 3, 4, 5])
            income = st.number_input("Gaji Bulanan ($)", 1000, 50000, 5000)
            overtime = st.selectbox("Lembur (OverTime)", ["No", "Yes"])
            travel = st.selectbox("Perjalanan Dinas", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

        with c3:
            st.markdown("### ⏱️ Riwayat & Kepuasan")
            total_working_years = st.number_input("Total Pengalaman Kerja (Thn)", 0, 40, 5)
            years_at_company = st.number_input("Lama di Perusahaan (Thn)", 0, 40, 3)
            years_in_role = st.number_input("Lama di Posisi Ini (Thn)", 0, 40, 2)
            
            st.markdown("---")
            st.caption("Skala 1 (Rendah) - 4 (Tinggi)")
            js = st.slider("Kepuasan Kerja", 1, 4, 3)
            es = st.slider("Kepuasan Lingkungan", 1, 4, 3)
            rs = st.slider("Kepuasan Hubungan", 1, 4, 3)
            ji = st.slider("Keterlibatan Kerja", 1, 4, 3)

        with st.expander("➕ Detail Tambahan (Opsional)", expanded=False):
             ec1, ec2 = st.columns(2)
             with ec1:
                 num_comp = st.number_input("Jml Perusahaan Sebelumnya", 0, 10, 1)
                 stock = st.selectbox("Level Opsi Saham (0-3)", [0, 1, 2, 3])
                 training = st.number_input("Training Tahun Lalu (Kali)", 0, 6, 2)
             with ec2:
                 wlb = st.slider("Work Life Balance", 1, 4, 3)
                 years_promo = st.number_input("Tahun Sejak Promosi Terakhir", 0, 40, 1)
                 years_manager = st.number_input("Tahun Bersama Manajer Saat Ini", 0, 40, 2)

        submit_btn = st.form_submit_button("🔍 Analisis Risiko Sekarang", use_container_width=True)
    
    if submit_btn:
        if model is None or feature_names is None:
            st.error("⚠️ Model belum dimuat. Pastikan file 'catboost_optimized.cbm' dan 'feature_names.pkl' ada.")
        else:
            data = {
                'Age': age, 'MonthlyIncome': income, 'TotalWorkingYears': total_working_years, 
                'OverTime': overtime, 'Department': dept, 'JobRole': role, 
                'MaritalStatus': marital_status, 'DistanceFromHome': distance, 
                'Education': education, 'EducationField': edu_field, 
                'BusinessTravel': travel, 'StockOptionLevel': stock, 
                'JobLevel': job_level, 'JobSatisfaction': js, 
                'EnvironmentSatisfaction': es, 'RelationshipSatisfaction': rs, 
                'JobInvolvement': ji, 'NumCompaniesWorked': num_comp, 
                'TrainingTimesLastYear': training, 'WorkLifeBalance': wlb, 
                'YearsAtCompany': years_at_company, 'YearsInCurrentRole': years_in_role, 
                'YearsSinceLastPromotion': years_promo, 'YearsWithCurrManager': years_manager
            }
            
            df_pred = pd.DataFrame([data])
            
            df_pred['TotalSatisfaction'] = (
                df_pred['JobSatisfaction'] + df_pred['EnvironmentSatisfaction'] + 
                df_pred['RelationshipSatisfaction'] + df_pred['JobInvolvement']
            )
            
            years_at_comp_safe = df_pred['YearsAtCompany'].replace(0, 0.1)
            total_working_safe = df_pred['TotalWorkingYears'].replace(0, 1)
            age_safe = df_pred['Age'].replace(0, 18) 
            
            df_pred['CareerStability'] = df_pred['YearsInCurrentRole'] / years_at_comp_safe
            df_pred['LoyaltyRatio'] = df_pred['YearsAtCompany'] / total_working_safe
            df_pred['IncomePerAge'] = df_pred['MonthlyIncome'] / age_safe
            
            try:
                for col in feature_names:
                    if col not in df_pred.columns:
                        df_pred[col] = 0 
                
                df_final = df_pred[feature_names]
                
                probabilitas = model.predict_proba(df_final)[0][1]
                
                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.metric("Probabilitas Attrition", f"{probabilitas:.1%}")
                
                with col_res2:
                    if probabilitas >= 0.279: 
                        st.error("🔴 **BERISIKO TINGGI (HIGH RISK)**")
                        st.write("Karyawan ini memiliki kemungkinan besar untuk meninggalkan perusahaan. Disarankan intervensi segera.")
                    else:
                        st.success("🟢 **AMAN (LOW RISK)**")
                        st.write("Karyawan ini diprediksi loyal dan tidak berisiko keluar dalam waktu dekat.")
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")

st.markdown("---")
st.caption("© 2025 Kelompok 9 - Final Project RSBP")