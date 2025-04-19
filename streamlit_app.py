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
st.title("🧾 Prediksi Persetujuan Pinjaman")

# Input form
st.subheader("Masukkan Data Calon Peminjam:")

person_age = st.number_input("Umur", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
person_education = st.selectbox("Pendidikan", ["High School", "Bachelor", "Master"])
person_income = st.number_input("Pendapatan per Tahun", min_value=0.0, value=50000.0)
person_emp_exp = st.slider("Pengalaman Kerja (tahun)", 0, 40, 5)
loan_amnt = st.number_input("Jumlah Pinjaman", min_value=0.0, value=10000.0)
loan_int_rate = st.number_input("Suku Bunga Pinjaman (%)", value=10.5)
loan_percent_income = loan_amnt / (person_income + 1e-6)
cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", value=3.0)
credit_score = st.number_input("Skor Kredit (300-850)", min_value=300, max_value=850, value=600)
previous_loan_defaults_on_file = st.selectbox("Riwayat Gagal Bayar", ["No", "Yes"])
loan_intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"])
home_ownership = st.selectbox("Status Tempat Tinggal", ["OWN", "RENT", "OTHER", "MORTGAGE"])

# Map gender
gender_map = {"Laki-laki": "male", "Perempuan": "female"}

# Prepare input dict
input_data = {
    'person_age': person_age,
    'person_gender': 1 if gender_map[person_gender] == 'male' else 0,
    'person_education': person_education,
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': 1 if previous_loan_defaults_on_file == "Yes" else 0
}

# One-hot encode loan_intent
for intent in ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"]:
    input_data[f"loan_intent_{intent}"] = 1 if loan_intent == intent else 0

# One-hot encode home_ownership
for ho in ["OTHER", "OWN", "RENT", "MORTGAGE"]:
    input_data[f"person_home_ownership_{ho}"] = 1 if home_ownership == ho else 0

# Encode education (optional: bisa juga pakai one-hot)
edu_map = {"High School": 0, "Bachelor": 1, "Master": 2}
input_data["person_education"] = edu_map[person_education]

# Predict
if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    result = model.predict(input_df)[0]
    st.success(f"🎯 Hasil Prediksi: {'DISETUJUI' if result == 1 else 'DITOLAK'}")

# Tambahkan test case
st.sidebar.header("💡 Test Case")
if st.sidebar.button("Test Case 1"):
    st.write("🔹 Laki-laki, Master, Pendapatan: 70K, Pinjaman: 20K, Kredit Baik, Tujuan: DEBTCONSOLIDATION")
if st.sidebar.button("Test Case 2"):
    st.write("🔹 Perempuan, SMA, Pendapatan: 25K, Pinjaman: 15K, Kredit Buruk, Tujuan: MEDICAL")

