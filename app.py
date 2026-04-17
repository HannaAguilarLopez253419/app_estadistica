import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


st.title("App de Estadística distribuciones de probabilidad y prueba de hipotesis")
st.write("Bienvenido a la aplicación")
st.header("Carga de datos")
archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])
if archivo:
    df = pd.read_csv(archivo)
    st.write("Vista previa de los datos:")
    st.write(df.head())
if 'df' in locals():

    st.subheader("Selección de variable")

    columna = st.selectbox("Selecciona una variable", df.columns)

    datos = df[columna]

    # Validación
    if datos.dtype != 'int64' and datos.dtype != 'float64':
        st.warning("Selecciona una variable numérica")
    else:
        st.success("Variable válida para análisis")
     st.subheader("Histograma")

fig, ax = plt.subplots()
sns.histplot(datos, kde=True, ax=ax)

st.pyplot(fig)   