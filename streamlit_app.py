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

# ========================
# 🌿 Judul Aplikasi
# ========================
st.title("📊 Prediksi Persetujuan Pinjaman")

# ========================
# 📝 Form Input Pengguna
# ========================
st.subheader("Masukkan Data Calon Peminjam:")

person_age = st.number_input("Usia (person_age)", min_value=18, max_value=100, value=30)

person_gender = st.selectbox("Jenis Kelamin (person_gender)", ["female", "male"])

person_education = st.selectbox(
    "Tingkat Pendidikan (person_education)",
    ["High School", "Bachelor", "Master", "Associate", "Doctorate"]
)

person_income = st.number_input("Pendapatan Tahunan (person_income)", min_value=0.0, value=50000.0)

person_emp_exp = st.slider("Lama Pengalaman Kerja (tahun) (person_emp_exp)", 0, 40, 5)

person_home_ownership = st.selectbox(
    "Status Kepemilikan Tempat Tinggal (person_home_ownership)",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_amnt = st.number_input("Jumlah Pinjaman (loan_amnt)", min_value=0.0, value=10000.0)

loan_intent = st.selectbox(
    "Tujuan Pinjaman (loan_intent)",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

loan_int_rate = st.number_input("Suku Bunga Pinjaman (%) (loan_int_rate)", value=10.5)

loan_percent_income = loan_amnt / (person_income + 1e-6)

cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun) (cb_person_cred_hist_length)", value=3.0)

credit_score = st.number_input("Skor Kredit (credit_score)", min_value=300, max_value=850, value=600)

previous_loan_defaults_on_file = st.selectbox(
    "Riwayat Gagal Bayar Sebelumnya (previous_loan_defaults_on_file)", ["No", "Yes"]
)

# ========================
# 🔀 Encoding & One-Hot
# ========================
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

# One-hot encode loan_intent
loan_intent_categories = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"]
for intent in loan_intent_categories:
    input_data[f"loan_intent_{intent}"] = 1 if loan_intent == intent else 0

# One-hot encode home_ownership
home_ownership_categories = ["RENT", "OWN", "OTHER"]
for ho in home_ownership_categories:
    input_data[f"person_home_ownership_{ho}"] = 1 if person_home_ownership == ho else 0

# ========================
# 🧐 Prediksi
# ========================
if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    result = model.predict(input_df)[0]
    st.success(f"📈 Hasil Prediksi: {'DISETUJUI ✅' if result == 1 else 'DITOLAK ❌'}")

# ========================
# 🧪 Test Case Samping
# ========================
st.sidebar.header("💡 Test Case")
if st.sidebar.button("Test Case 1"):
    st.write("🔹 Jenis Kelamin: male, Pendidikan: Master, Pendapatan: 70K, Pinjaman: 20K, Kredit: 720, Tujuan: DEBTCONSOLIDATION")

if st.sidebar.button("Test Case 2"):
    st.write("🔹 Jenis Kelamin: female, Pendidikan: High School, Pendapatan: 25K, Pinjaman: 15K, Kredit: 580, Tujuan: MEDICAL")
