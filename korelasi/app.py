import streamlit as st
import pandas as pd
from scipy.stats import pearsonr
import plotly.express as px
import os
import google.generativeai as genai
from pathlib import Path


st.markdown("""
    <style>
    .stButton > button {
    background-color: #4CAF50;
    border: 1px;
    }

    .stButton > button:hover {
    background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Dashboard Analisis Korelasi",
                   page_icon="📊",
                   layout="wide")
st.title("Dashboard Analisis Korelasi")

with st.sidebar:
    st.subheader("📌 Tentang Dashboard")

    st.markdown("""
    Dashboard ini digunakan untuk menganalisis hubungan antar variabel numerik, 
    menghitung korelasi Pearson, menampilkan visualisasi hubungan data, dan menentukan signifikansi statistik
    """)

    st.subheader("📑 Upload Dataset")
    file1 = st.file_uploader("Upload CSV atau Excel (Data 1)", type=["csv", "xlsx"])
    file2 = st.file_uploader("Upload CSV atau Excel (Data 2)", type=["csv", "xlsx"])
# -------------------------
# Gemini helpers
# -------------------------
def get_gemini_client():
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        key = None

    key = key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None

    genai.configure(api_key=key)
    return genai

def corr_strength(r: float) -> str:
    ar = abs(r)
    if ar < 0.10:
        return "Sangat Lemah"
    elif ar < 0.30:
        return "Lemah"
    elif ar < 0.50:
        return "Sedang"
    elif ar < 0.70:
        return "Kuat"
    else:
        return "Sangat Kuat"

def generate_ai_insight(client, context: dict) -> str:
    if client is None:
        return "⚠️ Insight AI belum aktif (API key belum diset)."

    prompt = f"""
Kamu adalah analis statistik yang ringkas dan jelas.

Buat insight sederhana dan profesional dengan format ini saja:

HASIL KORELASI
- Metode: {context['metode']}
- Koefisien: {context['koefisien']}
- P-value: {context['p_value']}
- Kekuatan: {context['kekuatan']}
- Status: {context['status']}

KUALITAS DATA
- Jumlah data valid: {context['n_valid']}
- Missing data: {context['missing']}
- Catatan: {context['catatan']}

KESIMPULAN
Jelaskan singkat apakah ada bukti hubungan statistik antara Data 1 dan Data 2.
Tegaskan bahwa korelasi bukan sebab-akibat.

NEXT STEP
Berikan satu rekomendasi langkah lanjut yang paling relevan (1 kalimat).

Gunakan Bahasa Indonesia yang jelas dan singkat.
"""
    model = client.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)
    return response.text.strip()

# -------------------------
# Upload file
# -------------------------

def read_file(f):
    return pd.read_csv(f) if f.name.endswith("csv") else pd.read_excel(f)

if file1 and file2:
    try:
        df1 = read_file(file1)
        df2 = read_file(file2)

        # Cek apakah jumlah baris sama
        if len(df1) != len(df2):
            st.error("Jumlah baris Data 1 dan Data 2 tidak sama. Pastikan jumlah baris dan kolom antar 2 data sesuai.")
            st.stop()

        st.subheader("📝 Ringkasan Korelasi")
        col = st.columns((0.8, 2), gap='medium')
        with col[0]:
            column_x = st.selectbox("Pilih Kolom untuk X (Data 1)", df1.columns)
            column_y = st.selectbox("Pilih Kolom untuk Y (Data 2)", df2.columns)

            # Paksa numeric + buang NaN pasangan
            x = pd.to_numeric(df1[column_x], errors="coerce")
            y = pd.to_numeric(df2[column_y], errors="coerce")
            valid = pd.DataFrame({"x": x, "y": y}).dropna()
        
        with col[1]:
            if len(valid) < 2:
                st.error("Jumlah data yang bisa dianalisis terlalu sedikit setelah menghapus data yang kosong atau tidak valid (minimal 2 data).")
                st.info(f"Data valid dipakai: {len(valid)} dari {len(df1)} baris. "
                    f"Missing/invalid total: {(x.isna().sum() + y.isna().sum())}")
                st.stop()

            # Korelasi Pearson
            correlation, p_value = pearsonr(valid["x"], valid["y"])
            strength = corr_strength(correlation)

            colm = st.columns((0.8, 0.8, 1), gap='small')
            with colm[0]:
                st.container(border=True).metric("Koefisien (r)", f"{correlation:.3f}")
            with colm[1]:
                st.container(border=True).metric("P-value", f"{p_value:.4f}")
            with colm[2]:
                st.container(border=True).metric("Kekuatan", strength)

            if p_value < 0.05:
                st.success("✅ Korelasi signifikan secara statistik (α = 0.05)")
            else:
                st.warning("⚠️ Korelasi tidak signifikan secara statistik (α = 0.05)")

        # Scatter plot
        fig = px.scatter(
            valid, x="x", y="y",
            labels={"x": column_x, "y": column_y},
        )
        st.subheader("📈 Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # Insight AI
        # -------------------------
        st.subheader("🤖 Analisis AI")

        client = get_gemini_client()

        context = {
            "metode": "Pearson",
            "koefisien": round(float(correlation), 3),
            "p_value": round(float(p_value), 4),
            "kekuatan": strength,
            "status": "Signifikan" if p_value < 0.05 else "Tidak Signifikan",
            "n_valid": int(len(valid)),
            "missing": int(x.isna().sum() + y.isna().sum()),
            "catatan": "Asumsi: Data 1 dan Data 2 sejajar berdasarkan urutan baris. Jika ada ID/tanggal, lebih valid merge by key."
        }

        if st.button("Generate AI"):
            with st.spinner("AI sedang menganalisis..."):
                insight = generate_ai_insight(client, context)
            st.markdown(insight)

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memproses file: {e}")
else:
    st.info("📌 Silakan unggah dua dataset pada sidebar untuk melihat hasil analisis.")
