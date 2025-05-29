import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuración general
st.set_page_config(layout="wide")
st.title("Dashboard de Préstamos - Ximple")

# Subida del archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Filtros en la barra lateral
    st.sidebar.header("Filtros")

    if "IssueYear" in df.columns:
        years = sorted(df["IssueYear"].dropna().unique())
        selected_years = st.sidebar.multiselect("Selecciona Año(s):", years, default=years)
        df = df[df["IssueYear"].isin(selected_years)]

    if "customer_region" in df.columns:
        regiones = sorted(df["customer_region"].dropna().unique())
        selected_regions = st.sidebar.multiselect("Selecciona Región(es):", regiones, default=regiones)
        df = df[df["customer_region"].isin(selected_regions)]

    if "canal_ajustado" in df.columns:
        canales = sorted(df["canal_ajustado"].dropna().unique())
        selected_channels = st.sidebar.multiselect("Selecciona Canal(es):", canales, default=canales)
        df = df[df["canal_ajustado"].isin(selected_channels)]

    # Vista previa
    st.subheader("Vista Previa de los Datos Filtrados")
    st.dataframe(df.head())

    # Visualización 1
    st.markdown("---")
    st.subheader("Participación de Préstamos por Canal")
    if "canal_ajustado" in df.columns:
        canal_counts = df["canal_ajustado"].value_counts().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        canal_counts.plot(kind="bar", ax=ax1)
        ax1.set_title("Número de Préstamos por Canal")
        ax1.set_xlabel("Canal")
        ax1.set_ylabel("Cantidad")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

    # Visualización 2
    st.markdown("---")
    st.subheader("Participación de Préstamos por Región")
    if "customer_region" in df.columns:
        region_counts = df["customer_region"].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)

    # Visualización 3
    st.markdown("---")
    st.subheader("Distribución de Préstamos por Región")
    if "customer_region" in df.columns:
        fig3, ax3 = plt.subplots()
        region_counts.sort_values().plot(kind="barh", ax=ax3)
        ax3.set_title("Distribución de Préstamos por Región")
        ax3.set_xlabel("Cantidad")
        ax3.set_ylabel("Región")
        st.pyplot(fig3)

    # Visualización 4
    st.markdown("---")
    st.subheader("Monto Promedio por Cliente y Tipo de Préstamo")
    if all(col in df.columns for col in ['LoanAmount', 'RecipientType', 'LoanType']):
        avg_amount = df.groupby(['RecipientType', 'LoanType'])['LoanAmount'].mean().reset_index()
        pivot_table = avg_amount.pivot(index='RecipientType', columns='LoanType', values='LoanAmount')
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        pivot_table.plot(kind='bar', ax=ax4)
        ax4.set_title("Monto Promedio del Préstamo")
        ax4.set_ylabel("Monto Promedio ($)")
        ax4.set_xlabel("Tipo de Cliente")
        ax4.legend(title="Tipo de Préstamo")
        st.pyplot(fig4)
    else:
        st.info("No se encontraron columnas LoanAmount, RecipientType o LoanType para esta visualización.")
else:
    st.warning("Por favor sube un archivo CSV para comenzar.")
