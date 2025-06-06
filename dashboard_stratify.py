import streamlit as st
import pandas as pd
import joblib
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
from streamlit_bokeh import streamlit_bokeh

# ============================
# CONFIGURACIÓN DEL DASHBOARD
# ============================
st.set_page_config(
    page_title="Ximple Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# ESTILOS PERSONALIZADOS
# ============================
st.markdown("""
<style>
    body, .stApp {
        background-color: #ffffff;
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .main { background-color: #ffffff; }
    .block-container { padding: 1rem 2rem; }
    .metric-label, .metric-value {
        color: #000000 !important;
    }
    .stButton>button {
        background-color: #bfa14c;
        color: white;
    }
    .stDownloadButton>button {
        background-color: #bfa14c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# CARGA DEL MODELO Y DATOS
# ============================
@st.cache_data
def load_model():
    return joblib.load("modelo_final.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("df_merged_clusters.csv")

modelo = load_model()
df = load_data()

# ============================
# SIDEBAR – FILTRO POR CLUSTER
# ============================
st.sidebar.title("Filtros")

# Diccionario de nombres descriptivos
cluster_names = {
    0: "Aliadas con pagos frecuentes y pocos atrasos",
    1: "Aliadas con alto volumen y alta morosidad",
    2: "Aliadas de baja actividad crediticia"
}

# Generar la lista para el selectbox
cluster_display = [f"{k} – {v}" for k, v in cluster_names.items()]
cluster_str = st.sidebar.selectbox("Selecciona el tipo de Aliada (cluster):", cluster_display)

# Extraer número del cluster
cluster_sel = int(cluster_str.split(" – ")[0])
df_cluster = df[df['cluster_kmeans'] == cluster_sel]

# ✅ Convertimos intensive_use a str
df_cluster["intensive_use"] = df_cluster["intensive_use"].astype(str)

# ============================
# TÍTULO PRINCIPAL
# ============================
st.title("Ximple – Dashboard de Comportamiento Crediticio")
st.markdown("Este dashboard presenta insights clave del comportamiento crediticio de las Aliadas segmentadas por tipo de uso.")
st.markdown(f"### {cluster_names[cluster_sel]}")  # Subtítulo dinámico

# ============================
# MÉTRICAS GENERALES
# ============================
st.subheader("Métricas del Cluster")
col1, col2, col3 = st.columns(3)
col1.metric("Total Aliadas", len(df_cluster))
col2.metric("% Intensivas", f"{100 * df_cluster['intensive_use'].astype(int).mean():.1f}%")
col3.metric("Prom. días entre préstamos", f"{df_cluster['dias_promedio'].mean():.1f} días")

# ============================
# GRÁFICAS BOKEH
# ============================

# 1. Tipo de Préstamo por Cliente
st.subheader("Tipo de Préstamo por Cliente")
tipo_prestamo = df_cluster.groupby(["LoanType", "RecipientType"]).size().reset_index(name="count")
src1 = ColumnDataSource(tipo_prestamo)
fig1 = figure(x_range=tipo_prestamo["LoanType"].unique(), height=350, title="Distribución de tipo de préstamo",
              toolbar_location=None, tools="")
fig1.vbar(x="LoanType", top="count", width=0.7, source=src1, legend_field="RecipientType",
          fill_color=factor_cmap('LoanType', palette=Category10[10], factors=list(tipo_prestamo["LoanType"].unique())))
fig1.xgrid.grid_line_color = None
fig1.legend.orientation = "horizontal"
streamlit_bokeh(fig1, use_container_width=True, key="grafica1")

# 2. Frecuencia vs. Simultaneidad
st.subheader("Frecuencia vs. Simultaneidad")
src2 = ColumnDataSource(df_cluster)
fig2 = figure(height=350, title="Días promedio vs. Préstamos activos")
fig2.circle(x="dias_promedio", y="prestamos_outstanding", size=8, source=src2,
            color=factor_cmap('intensive_use', ['#bfa14c', '#000000'], ['0', '1']), legend_field="intensive_use")
fig2.xaxis.axis_label = "Días promedio"
fig2.yaxis.axis_label = "Préstamos activos"
streamlit_bokeh(fig2, use_container_width=True, key="grafica2")

# 3. Mora por Tipo de Cliente
st.subheader("Mora por Tipo de Cliente")
tipo_mora = df_cluster.groupby("RecipientType")["cuotas_mora"].mean().reset_index()
src3 = ColumnDataSource(tipo_mora)
fig3 = figure(x_range=tipo_mora["RecipientType"], height=350, title="Prom. Mora por Cliente")
fig3.vbar(x="RecipientType", top="cuotas_mora", width=0.5, source=src3, color="#bfa14c")
fig3.xgrid.grid_line_color = None
streamlit_bokeh(fig3, use_container_width=True, key="grafica3")

# 4. Distribución por Región
st.subheader("Distribución por Región")
region = df_cluster["customer_region"].value_counts().reset_index()
region.columns = ["Region", "Total"]
src4 = ColumnDataSource(region)
fig4 = figure(x_range=region["Region"], height=350, title="Clientes por región")
fig4.vbar(x="Region", top="Total", width=0.6, source=src4, color="#bfa14c")
streamlit_bokeh(fig4, use_container_width=True, key="grafica4")

# ============================
# PREDICCIÓN SOBRE NUEVOS DATOS
# ============================
st.subheader("Predicción de Clientes Intensivos")
file = st.file_uploader("Carga un archivo CSV para predecir (con columnas requeridas)", type="csv")

if file:
    new_data = pd.read_csv(file)
    try:
        predictions = modelo.predict(new_data)
        new_data['Prediccion_Intensive'] = predictions
        st.success("✅ Predicciones realizadas")
        st.dataframe(new_data)
        st.download_button("Descargar resultados", new_data.to_csv(index=False), file_name="predicciones.csv")
    except Exception as e:
        st.error(f"❌ Error al predecir: {e}")

# ============================
# PIE DE PÁGINA
# ============================
st.markdown("---")
st.caption("Ximple Dashboard · Aliadas Credit Behavior · Proyecto final de consultoría")
