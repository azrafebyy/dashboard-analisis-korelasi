import streamlit as st
import pandas as pd
from scipy.stats import pearsonr
import plotly.express as px
import os
import google.generativeai as genai
from pathlib import Path

# Load custom CSS
def load_css(filename="style.css"):
    base_dir = Path(__file__).parent
    css_path = base_dir / filename

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"{filename} belum ditemukan di folder korelasi")

# Panggil untuk load CSS
load_css()

st.set_page_config(page_title="Dashboard Analisis Korelasi", page_icon="📊")
st.title("Dashboard Analisis Korelasi")
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
file1 = st.file_uploader("Upload CSV atau Excel (Data 1)", type=["csv", "xlsx"])
file2 = st.file_uploader("Upload CSV atau Excel (Data 2)", type=["csv", "xlsx"])

def read_file(f):
    return pd.read_csv(f) if f.name.endswith("csv") else pd.read_excel(f)

if file1 and file2:
    try:
        df1 = read_file(file1)
        df2 = read_file(file2)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📥 Input Data")

        c1, c2 = st.columns(2)
        with c1:
            column_x = st.selectbox("Pilih Kolom untuk X (Data 1)", df1.columns)
        with c2:
            column_y = st.selectbox("Pilih Kolom untuk Y (Data 2)", df2.columns)

        st.markdown('</div>', unsafe_allow_html=True)

        # Cek apakah jumlah baris sama
        if len(df1) != len(df2):
            st.error("Jumlah baris Data 1 dan Data 2 tidak sama. Pastikan jumlah baris dan kolom antar 2 data sesuai.")
            st.stop()

        # Paksa numeric + buang NaN pasangan
        x = pd.to_numeric(df1[column_x], errors="coerce")
        y = pd.to_numeric(df2[column_y], errors="coerce")
        valid = pd.DataFrame({"x": x, "y": y}).dropna()

        if len(valid) < 2:
            st.error("Jumlah data yang bisa dianalisis terlalu sedikit setelah menghapus data yang kosong atau tidak valid (minimal 2 data).")
            st.stop()


        # Korelasi Pearson
        correlation, p_value = pearsonr(valid["x"], valid["y"])
        strength = corr_strength(correlation)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Ringkasan Korelasi")
        m1, m2, m3 = st.columns(3)
        m1.metric("Koefisien (r)", f"{correlation:.3f}")
        m2.metric("P-value", f"{p_value:.4f}")
        m3.metric("Kekuatan", strength)

        st.write("Status:", "✅ Signifikan" if p_value < 0.05 else "⚠️ Tidak signifikan")
        st.markdown('</div>', unsafe_allow_html=True)

        st.info(f"Data valid dipakai: {len(valid)} dari {len(df1)} baris. "
                f"Missing/invalid total: {(x.isna().sum() + y.isna().sum())}")

        # Scatter plot
        fig = px.scatter(
            valid, x="x", y="y",
            labels={"x": column_x, "y": column_y},
            title="Scatter Plot (Data valid)"
        )
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------
        # Insight AI
        # -------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
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

        st.write("DEBUG: Sampai Analisis AI ✅")
        st.write("DEBUG: Gemini client aktif?", client is not None)

        if st.button("Generate AI"):
            with st.spinner("AI sedang menganalisis..."):
                insight = generate_ai_insight(client, context)
            st.markdown(insight)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memproses file: {e}")
else:
    st.info("Upload dua file untuk mulai analisis.")

