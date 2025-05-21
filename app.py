import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Analisis Kekerasan Anak di Sumatera Utara", layout="wide")
st.title("Analisis Kekerasan Anak di Sumatera Utara")

# Upload file Excel
data_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if data_file is not None:
    # Baca data
    data = pd.read_excel(data_file)

    # Tampilkan 5 data teratas
    st.subheader("5 Data Teratas")
    st.write(data.head())

    # Info Data
    st.subheader("Info Dataset")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Statistik Deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(data.describe())

    # Cek missing values
    st.subheader("Jumlah Missing Values per Kolom")
    st.write(data.isnull().sum())

    # Visualisasi Korelasi
    st.subheader("Heatmap Korelasi (Kolom Numerik Saja)")
    numeric_data = data.select_dtypes(include=['number'])
    if not numeric_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Tidak ada kolom numerik untuk ditampilkan dalam heatmap.")

    # Distribusi Data Numerik
    st.subheader("Distribusi Setiap Kolom Numerik")
    for column in numeric_data.columns:
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(f"Distribusi: {column}")
        st.pyplot(fig)
else:
    st.warning("Silakan upload file Excel terlebih dahulu.")
