import pandas as pd
import streamlit as st

st.title("App de Estadística distribuciones de probabilidad y prueba de hipotesis")
st.write("Bienvenido a la aplicación")
st.header("Carga de datos")
archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])
if archivo:
    df = pd.read_csv(archivo)
    st.write("Vista previa de los datos:")
    st.write(df.head())
