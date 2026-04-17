import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import json

# ─────────────────────────────────────────────
#  CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Estadística Interactiva",
    page_icon="📊",
    layout="wide"
)

st.title("📊 App de Estadística: Distribuciones y Prueba de Hipótesis")
st.write("Carga tus datos o genera una muestra sintética para analizar distribuciones y ejecutar pruebas estadísticas.")

# ─────────────────────────────────────────────
#  MÓDULO 1 – CARGA DE DATOS
# ─────────────────────────────────────────────
st.header("1. Carga de datos")

modo = st.radio("Fuente de datos", ["Subir CSV", "Generar datos sintéticos"])

df = None

if modo == "Subir CSV":
    archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.success("Archivo cargado correctamente.")
        st.dataframe(df.head())

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        distribucion = st.selectbox("Distribución", ["Normal", "Sesgada (log-normal)", "Uniforme"])
    with col2:
        n_sintetico = st.slider("Tamaño de muestra (n)", 30, 1000, 100)
    with col3:
        semilla = st.number_input("Semilla aleatoria", value=42, step=1)

    np.random.seed(int(semilla))
    if distribucion == "Normal":
        datos_sint = np.random.normal(loc=50, scale=10, size=n_sintetico)
    elif distribucion == "Sesgada (log-normal)":
        datos_sint = np.random.lognormal(mean=3.5, sigma=0.5, size=n_sintetico)
    else:
        datos_sint = np.random.uniform(low=20, high=80, size=n_sintetico)

    df = pd.DataFrame({"variable_sintetica": datos_sint})
    st.info(f"Datos sintéticos generados: {n_sintetico} observaciones ({distribucion}).")
    st.dataframe(df.head())

    # ─────────────────────────────────────────────
#  MÓDULO 2 – SELECCIÓN DE VARIABLE Y VISUALIZACIÓN
# ─────────────────────────────────────────────
if df is not None:
    st.header("2. Selección de variable y visualización")

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columnas_numericas:
        st.error("El archivo no contiene columnas numéricas.")
        st.stop()

    columna = st.selectbox("Selecciona una variable numérica", columnas_numericas)
    datos = df[columna].dropna()

    st.subheader("Estadísticos descriptivos")
    desc = datos.describe()
    skewness = datos.skew()
    kurtosis = datos.kurtosis()
    IQR_val  = desc['75%'] - desc['25%']

    sesgo_label = "≈ simétrico" if abs(skewness) < 0.5 else (
        "Positivo (→ derecha)" if skewness > 0 else "Negativo (← izquierda)"
    )
    kurt_label = "Leptocúrtica (pico alto)" if kurtosis > 0 else (
        "Mesocúrtica (normal)" if kurtosis == 0 else "Platocúrtica (pico bajo)"
    )

    tabla_stats = pd.DataFrame({
        "Estadístico": [
            "n (tamaño de muestra)",
            "Media (x̄)",
            "Mediana (Q2 / 50%)",
            "Desviación estándar (s)",
            "Mínimo",
            "Máximo",
            "Q1 (25%)",
            "Q3 (75%)",
            "Rango IQR (Q3 − Q1)",
            "Sesgo (skewness)",
            "Curtosis (exceso)",
        ],
        "Valor": [
            f"{int(desc['count'])}",
            f"{desc['mean']:.4f}",
            f"{desc['50%']:.4f}",
            f"{desc['std']:.4f}",
            f"{desc['min']:.4f}",
            f"{desc['max']:.4f}",
            f"{desc['25%']:.4f}",
            f"{desc['75%']:.4f}",
            f"{IQR_val:.4f}",
            f"{skewness:.4f}",
            f"{kurtosis:.4f}",
        ],
        "Interpretación": [
            "Observaciones válidas",
            "Promedio aritmético",
            "Valor central de la distribución",
            "Dispersión respecto a la media",
            "Valor más pequeño observado",
            "Valor más grande observado",
            "25% de los datos están por debajo",
            "75% de los datos están por debajo",
            "Dispersión del 50% central",
            sesgo_label,
            kurt_label,
        ],
    })

    st.dataframe(tabla_stats, use_container_width=True, hide_index=True)

    st.divider()

    # ── GRÁFICAS ──
    st.subheader("Visualización de la distribución")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        plt.style.use("seaborn-whitegrid")

    PALETTE = {"hist": "#4C72B0", "kde": "#DD8452", "box": "#55A868"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Histograma + KDE
    sns.histplot(datos, kde=True, ax=axes[0],
                 color=PALETTE["hist"], edgecolor="white")
    axes[0].set_title("Histograma + KDE")
    axes[0].set_xlabel(columna)

    # KDE
    sns.kdeplot(datos, ax=axes[1], fill=True,
                color=PALETTE["kde"])
    axes[1].set_title("Densidad (KDE)")
    axes[1].set_xlabel(columna)

    # Boxplot
    sns.boxplot(y=datos, ax=axes[2],
                color=PALETTE["box"])
    axes[2].set_title("Boxplot")

    plt.tight_layout()
    st.pyplot(fig)