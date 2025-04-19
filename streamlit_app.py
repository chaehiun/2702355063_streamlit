# app.py
import streamlit as st
import pickle
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    with open("best_xgb_model_new.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title
st.title("Loan Approval Prediction")

# Input form
st.subheader("Masukkan Data Calon Peminjam:")
person_age = st.number_input("Usia (person_age)", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Jenis Kelamin (person_gender)", ["male", "female"])
person_education = st.selectbox("Pendidikan (person_education)", ["High School", "Bachelor", "Master","Associate", "Doctorate"])
person_income = st.number_input("Pendapatan per Tahun (person_income)", value=50000)
person_emp_exp = st.slider("Pengalaman Kerja (tahun) (person_emp_exp)", 0, 40, 5)
person_home_ownership = st.selectbox("Status Kepemilikan Tempat Tinggal (person_home_ownership)", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Jumlah Pinjaman (loan_amnt)", value=10000)
loan_intent = st.selectbox("Tujuan Pinjaman (loan_intent)", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_int_rate = st.number_input("Suku Bunga Pinjaman (%) (loan_int_rate)", value=10.5)
loan_percent_income = loan_amnt / (person_income + 1e-6)
cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun) (cb_person_cred_hist_length)", value=3)
credit_score = st.number_input("Skor Kredit (credit_score)", min_value=300, max_value=850, value=600)
previous_loan_defaults_on_file = st.selectbox("Riwayat Gagal Bayar (previous_loan_defaults_on_file)", ["No", "Yes"])

# Map categorical to numerical
gender_map = {"male": 1, "female": 0}
edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "Associate": 3, "Doctorate": 4}
default_map = {"No": 0, "Yes": 1}

loan_intents = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
home_ownerships = ["RENT", "OWN", "MORTGAGE", "OTHER"]

input_data = {
    'person_age': person_age,
    'person_gender': gender_map[person_gender],
    'person_education': edu_map[person_education],
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': default_map[previous_loan_defaults_on_file]
}

# One-hot encode loan_intent
for intent in loan_intents:
    input_data[f"loan_intent_{intent}"] = 1 if loan_intent == intent else 0

# One-hot encode home_ownership
for ho in home_ownerships:
    input_data[f"person_home_ownership_{ho}"] = 1 if person_home_ownership == ho else 0

if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])

    # 🔍 DEBUG: Cek fitur input yang kamu berikan ke model
    st.write("📋 Kolom input_df:", input_df.columns.tolist())
    
    # 🔍 DEBUG: Cek fitur yang dikenal oleh model (XGBoost booster)
    st.write("🧠 Kolom yang dikenal model:", model.get_booster().feature_names)

    # Prediksi
    result = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi: {'DISETUJUI' if result == 1 else 'DITOLAK'}")

# Tambahkan test case
st.sidebar.header("💡 Test Case")
if st.sidebar.button("Test Case 1"):
    st.write("🔹 Laki-laki, Sarjana, Pendapatan: 60K, Pinjaman: 10K, Skor Kredit: Baik, Tujuan: PERSONAL")
if st.sidebar.button("Test Case 2"):
    st.write("🔹 Perempuan, SMA, Pendapatan: 20K, Pinjaman: 15K, Skor Kredit: Buruk, Tujuan: MEDICAL")
