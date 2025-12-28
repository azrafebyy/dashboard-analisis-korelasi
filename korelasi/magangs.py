import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- Judul ---
st.title("Analisis Korelasi Penyakit vs Faktor")

# --- 1. Baca dataset ---
df_hiv = pd.read_excel(r"d:\diskominfo\korelasi\HIV1.xlsx")
df_dbd = pd.read_excel("DBD.xls")     # kolom: tahun, Kab_Kota, Jumlah, Jumlah meninggal
df_faktor = pd.read_excel("data_faktor.xlsx")  # kolom: tahun, Kab_Kota, pendapatan, pend

# --- 2. Gabungkan semua penyakit jadi satu dataframe ---
# rename biar konsisten
df_hiv_long = df_hiv.melt(id_vars=["tahun","Kab_Kota"], 
                          value_vars=["HIV","AIDS"], 
                          var_name="penyakit", 
                          value_name="jumlah")

df_dbd_long = df_dbd.melt(id_vars=["tahun","Kab_Kota"],
                          value_vars=["DBD","Meninggal DBD"], 
                          var_name="penyakit", 
                          value_name="jumlah")

# satukan
df_penyakit = pd.concat([df_hiv_long, df_dbd_long], ignore_index=True)

# gabungkan dengan faktor
df_all = pd.merge(df_penyakit, df_faktor, on=["tahun","Kab_Kota"], how="inner")

# --- 3. Sidebar pilihan user ---
st.sidebar.header("Filter Data")
tahun_list = sorted(df_all["tahun"].unique())
tahun_pilih = st.sidebar.selectbox("Pilih Tahun", tahun_list)

penyakit_list = df_all["penyakit"].unique().tolist()
penyakit_pilih = st.sidebar.selectbox("Pilih Penyakit", penyakit_list)

faktor_list = ["Pendapatan","Pendidikan","FASKES"]
faktor_pilih = st.sidebar.selectbox("Pilih Faktor", faktor_list)

# --- 4. Filter sesuai pilihan user ---
df_filtered = df_all[(df_all["tahun"] == tahun_pilih) & 
                     (df_all["penyakit"] == penyakit_pilih)]

st.write(f"### Data: {penyakit_pilih} vs {faktor_pilih} (Tahun {tahun_pilih})")
st.dataframe(df_filtered[["Kab_Kota","jumlah",faktor_pilih]])

# --- 5. Hitung korelasi Pearson ---
if df_filtered["jumlah"].notna().sum() > 1:
    corr, pval = pearsonr(df_filtered["jumlah"], df_filtered[faktor_pilih])
    st.write(f"**Korelasi Pearson = {corr:.2f} (p-value={pval:.4f})**")
    
    # Scatter plot
    fig, ax = plt.subplots()
    sns.regplot(x=df_filtered[faktor_pilih], y=df_filtered["jumlah"], ax=ax)
    ax.set_xlabel(faktor_pilih)
    ax.set_ylabel(f"Jumlah {penyakit_pilih}")
    ax.set_title(f"{penyakit_pilih} vs {faktor_pilih} ({tahun_pilih})")
    st.pyplot(fig)
else:
    st.warning("Data tidak cukup untuk menghitung korelasi.")

