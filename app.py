import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import json

st.set_page_config(
    page_title="Estadística Interactiva",
    page_icon="📊",
    layout="wide"
)
 
st.title("📊 App de Estadística: Distribuciones y Prueba de Hipótesis")
st.write("Carga tus datos o genera una muestra sintética para analizar distribuciones y ejecutar pruebas estadísticas.")
st.header("Carga de datos")

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
 
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Media", f"{desc['mean']:.4f}")
    c2.metric("Desv. estándar", f"{desc['std']:.4f}")
    c3.metric("Mínimo", f"{desc['min']:.4f}")
    c4.metric("Máximo", f"{desc['max']:.4f}")
    c5.metric("n", f"{int(desc['count'])}")
 
    c6, c7 = st.columns(2)
    c6.metric("Sesgo (skewness)", f"{skewness:.4f}")
    c7.metric("Curtosis (kurtosis)", f"{kurtosis:.4f}")