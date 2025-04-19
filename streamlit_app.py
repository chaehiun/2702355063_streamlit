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

# ========================
# 🏷️ Judul Aplikasi
# ========================
st.title("📊 Prediksi Persetujuan Pinjaman")

# ========================
# 📝 Form Input Pengguna
# ========================
st.subheader("Masukkan Data Calon Peminjam:")

person_age = st.number_input("a. Usia", min_value=18, max_value=100, value=30)

person_gender = st.selectbox("b. Jenis Kelamin", ["Laki-laki", "Perempuan"])

person_education = st.selectbox(
    "c. Tingkat Pendidikan",
    ["High School", "Bachelor", "Master"]
)

person_income = st.number_input("d. Pendapatan Tahunan", min_value=0.0, value=50000.0)

person_emp_exp = st.slider("e. Lama Pengalaman Kerja (tahun)", 0, 40, 5)

person_home_ownership = st.selectbox(
    "f. Status Kepemilikan Tempat Tinggal",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_amnt = st.number_input("g. Jumlah Pinjaman", min_value=0.0, value=10000.0)

loan_intent = st.selectbox(
    "h. Tujuan Pinjaman",
    ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"]
)

loan_int_rate = st.number_input("i. Suku Bunga Pinjaman (%)", value=10.5)

loan_percent_income = loan_amnt / (person_income + 1e-6)

cb_person_cred_hist_length = st.number_input("j. Lama Riwayat Kredit (tahun)", value=3.0)

credit_score = st.number_input("k. Skor Kredit (300-850)", min_value=300, max_value=850, value=600)

previous_loan_defaults_on_file = st.selectbox("l. Riwayat Gagal Bayar Sebelumnya", ["No", "Yes"])

# ========================
# 🔁 Encoding dan Input
# ========================
gender_map = {"Laki-laki": 1, "Perempuan": 0}
edu_map = {"High School": 0, "Bachelor": 1, "Master": 2}
default_map = {"No": 0, "Yes": 1}

input_data = {
    "person_age": person_age,
    "person_gender": gender_map[person_gender],
    "person_education": edu_map[person_education],
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": default_map[previous_loan_defaults_on_file],
}

# One-hot encode untuk loan_intent
for intent in ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"]:
    input_data[f"loan_intent_{intent}"] = 1 if loan_intent == intent else 0

# One-hot encode untuk person_home_ownership
for ho in ["RENT", "OWN", "MORTGAGE", "OTHER"]:
    input_data[f"person_home_ownership_{ho}"] = 1 if person_home_ownership == ho else 0

# ========================
# 🧠 Prediksi
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
    st.write("🔹 Master, Laki-laki, Pendapatan: 70K, Pinjaman: 20K, Kredit Baik, Tujuan: DEBTCONSOLIDATION")

if st.sidebar.button("Test Case 2"):
    st.write("🔹 High School, Perempuan, Pendapatan: 25K, Pinjaman: 15K, Kredit Buruk, Tujuan: MEDICAL")
