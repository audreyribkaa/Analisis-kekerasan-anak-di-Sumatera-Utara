import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    roc_curve, auc, roc_auc_score, f1_score, precision_score, recall_score
)

st.title("Analisis Kasus Kekerasan di Provinsi Sumatera Utara - 2018")

# Upload file
uploaded_file = st.file_uploader("Upload file Excel: dataset siga.xlsx", type=["xlsx"])

if uploaded_file:
    data = pd.read_excel(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(data.head())

    st.write("Jumlah Baris dan Kolom:", data.shape)
    st.write("Jumlah Baris:", data.shape[0])
    st.write("Jumlah Kolom:", data.shape[1])

    # Info data
    st.subheader("Informasi Dataset")
    buffer = []
    data.info(buf=buffer)
    s = "\n".join(buffer)
    st.text(s)

    # Cek missing values
    st.subheader("Cek Missing Values")
    st.write(data.isnull().sum())

    # Visualisasi
    st.subheader("Visualisasi Korelasi")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())

    # Jika ingin melanjutkan preprocessing dan modeling, bisa ditambahkan di bawah ini
    st.subheader("(Opsional) Model & Prediksi")
    st.info("Bagian ini bisa ditambahkan model klasifikasi atau regresi sesuai kebutuhan.")
else:
    st.warning("Silakan upload file 'dataset siga.xlsx' terlebih dahulu.")
