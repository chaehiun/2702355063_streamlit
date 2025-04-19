import streamlit as st
import pickle
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    with open("best_xgb_model_new.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load urutan kolom saat training
@st.cache_resource
def load_feature_order():
    with open("feature_order.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
feature_order = load_feature_order()

st.title("📊 Prediksi Persetujuan Pinjaman")

st.subheader("Masukkan Data Calon Peminjam:")

person_age = st.number_input("Usia (person_age)", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Jenis Kelamin (person_gender)", ["female", "male"])
person_education = st.selectbox("Tingkat Pendidikan (person_education)", ["High School", "Bachelor", "Master", "Associate", "Doctorate"])
person_income = st.number_input("Pendapatan Tahunan (person_income)", min_value=0.0, value=50000.0)
person_emp_exp = st.slider("Lama Pengalaman Kerja (tahun) (person_emp_exp)", 0, 40, 5)
person_home_ownership = st.selectbox("Status Kepemilikan Tempat Tinggal (person_home_ownership)", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Jumlah Pinjaman (loan_amnt)", min_value=0.0, value=10000.0)
loan_intent = st.selectbox("Tujuan Pinjaman (loan_intent)", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_int_rate = st.number_input("Suku Bunga Pinjaman (%) (loan_int_rate)", value=10.5)
loan_percent_income = loan_amnt / (person_income + 1e-6)
cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun) (cb_person_cred_hist_length)", value=3.0)
credit_score = st.number_input("Skor Kredit (credit_score)", min_value=300, max_value=850, value=600)
previous_loan_defaults_on_file = st.selectbox("Riwayat Gagal Bayar Sebelumnya (previous_loan_defaults_on_file)", ["No", "Yes"])

# Encoding
input_data = {
    "person_age": person_age,
    "person_gender": 1 if person_gender == "male" else 0,
    "person_education": ["High School", "Bachelor", "Master", "Associate", "Doctorate"].index(person_education),
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": 1 if previous_loan_defaults_on_file == "Yes" else 0,
}

# One-hot encoding
loan_intents = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
for intent in loan_intents:
    input_data[f"loan_intent_{intent}"] = 1 if loan_intent == intent else 0

home_ownerships = ["RENT", "OWN", "MORTGAGE", "OTHER"]
for ho in home_ownerships:
    input_data[f"person_home_ownership_{ho}"] = 1 if person_home_ownership == ho else 0

# Buat dataframe dan urutkan kolomnya
input_df = pd.DataFrame([input_data])
input_df = input_df[feature_order]  # urutkan sesuai saat training

# Prediksi
if st.button("Prediksi"):
    result = model.predict(input_df)[0]
    st.success(f"📈 Hasil Prediksi: {'DISETUJUI ✅' if result == 1 else 'DITOLAK ❌'}")
