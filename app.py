import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ia_helper import iniciar_ia, consultar_ia

# ─────────────────────────────────────────────
#  CONFIGURACIÓN GENERAL  ← debe ir PRIMERO
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Estadística Interactiva",
    page_icon="📊",
    layout="wide"
)

# Configurar IA (la key está en ia_helper.py)
gemini_listo = iniciar_ia()
gemini_model = None

# Inicializar session_state para la respuesta de IA
if "respuesta_ia" not in st.session_state:
    st.session_state.respuesta_ia = None
if "error_ia" not in st.session_state:
    st.session_state.error_ia = None

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
    desc     = datos.describe()
    skewness = datos.skew()
    kurtosis = datos.kurtosis()
    IQR_val  = desc["75%"] - desc["25%"]

    sesgo_label = "≈ simétrico" if abs(skewness) < 0.5 else (
        "Positivo (→ derecha)" if skewness > 0 else "Negativo (← izquierda)"
    )
    kurt_label = "Leptocúrtica (pico alto)" if kurtosis > 0 else (
        "Mesocúrtica (normal)" if kurtosis == 0 else "Platocúrtica (pico bajo)"
    )

    tabla_stats = pd.DataFrame({
        "Estadístico": [
            "n (tamaño de muestra)", "Media (x̄)", "Mediana (Q2 / 50%)",
            "Desviación estándar (s)", "Mínimo", "Máximo",
            "Q1 (25%)", "Q3 (75%)", "Rango IQR (Q3 − Q1)",
            "Sesgo (skewness)", "Curtosis (exceso)",
        ],
        "Valor": [
            f"{int(desc['count'])}", f"{desc['mean']:.4f}", f"{desc['50%']:.4f}",
            f"{desc['std']:.4f}", f"{desc['min']:.4f}", f"{desc['max']:.4f}",
            f"{desc['25%']:.4f}", f"{desc['75%']:.4f}", f"{IQR_val:.4f}",
            f"{skewness:.4f}", f"{kurtosis:.4f}",
        ],
        "Interpretación": [
            "Observaciones válidas", "Promedio aritmético", "Valor central",
            "Dispersión respecto a la media", "Valor más pequeño", "Valor más grande",
            "25% de los datos por debajo", "75% de los datos por debajo",
            "Dispersión del 50% central", sesgo_label, kurt_label,
        ],
    })

    st.dataframe(tabla_stats, use_container_width=True, hide_index=True)
    st.divider()

    # ── GRÁFICAS ──
    st.subheader("Visualización de la distribución")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    PALETTE = {"hist": "#4C72B0", "kde": "#DD8452", "box": "#55A868"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.histplot(datos, kde=True, ax=axes[0], color=PALETTE["hist"], edgecolor="white")
    axes[0].set_title("Histograma + KDE")

    sns.kdeplot(datos, ax=axes[1], fill=True, color=PALETTE["kde"])
    axes[1].set_title("Densidad (KDE)")

    sns.boxplot(y=datos, ax=axes[2], color=PALETTE["box"])
    axes[2].set_title("Boxplot")

    plt.tight_layout()
    st.pyplot(fig)
    plt.style.use("default")
    st.divider()

    # ── ANÁLISIS AUTOMÁTICO ──
    st.subheader("Análisis de la Distribución")

    col_an1, col_an2, col_an3 = st.columns(3)

    with col_an1:
        st.write("**1. ¿La distribución parece normal?**")
        if abs(skewness) < 0.5:
            st.success("Probable")
        else:
            st.warning("Poco probable")

    with col_an2:
        st.write("**2. ¿Hay sesgo?**")
        if abs(skewness) < 0.5:
            st.info("Simétrica")
        elif skewness > 0:
            st.info("Positivo (Derecha)")
        else:
            st.info("Negativo (Izquierda)")

    with col_an3:
        z_scores     = np.abs(stats.zscore(datos))
        num_outliers = int(np.sum(z_scores > 3))
        st.write("**3. ¿Hay outliers?**")
        if num_outliers > 0:
            st.error(f"Sí, detectados: {num_outliers}")
        else:
            st.success("No detectados")

    # ─────────────────────────────────────────────
    #  MÓDULO 3 – PRUEBA DE HIPÓTESIS (Prueba Z)
    # ─────────────────────────────────────────────
    st.header("3. Prueba de hipótesis – Prueba Z")
    st.markdown(
        "Esta prueba asume **varianza poblacional conocida** y **n ≥ 30**. "
        "Evalúa si la media muestral difiere significativamente de un valor hipotético."
    )

    if len(datos) < 30:
        st.error(f"La muestra tiene solo {len(datos)} observaciones. Se requieren n ≥ 30 para la Prueba Z.")
    else:
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            mu0       = st.number_input("Valor hipotético de la media (μ₀)", value=float(round(datos.mean(), 2)))
            sigma_pob = st.number_input(
                "Desviación estándar poblacional (σ)",
                value=float(round(datos.std(), 4)),
                min_value=0.0001
            )
        with col_h2:
            alpha       = st.selectbox("Nivel de significancia (α)", [0.01, 0.05, 0.10], index=1)
            tipo_prueba = st.selectbox(
                "Tipo de prueba",
                ["Bilateral (H₁: μ ≠ μ₀)", "Cola izquierda (H₁: μ < μ₀)", "Cola derecha (H₁: μ > μ₀)"]
            )

        st.markdown(f"**H₀:** μ = {mu0}")
        if "Bilateral" in tipo_prueba:
            st.markdown(f"**H₁:** μ ≠ {mu0}")
        elif "izquierda" in tipo_prueba:
            st.markdown(f"**H₁:** μ < {mu0}")
        else:
            st.markdown(f"**H₁:** μ > {mu0}")

        # ── Cálculo Z ──
        n         = len(datos)
        x_bar     = datos.mean()
        error_std = sigma_pob / np.sqrt(n)
        Z_calc    = (x_bar - mu0) / error_std

        if "Bilateral" in tipo_prueba:
            p_value   = 2 * (1 - stats.norm.cdf(abs(Z_calc)))
            Z_critico = stats.norm.ppf(1 - alpha / 2)
            rechazar  = abs(Z_calc) > Z_critico
        elif "izquierda" in tipo_prueba:
            p_value   = stats.norm.cdf(Z_calc)
            Z_critico = stats.norm.ppf(alpha)
            rechazar  = Z_calc < Z_critico
        else:
            p_value   = 1 - stats.norm.cdf(Z_calc)
            Z_critico = stats.norm.ppf(1 - alpha)
            rechazar  = Z_calc > Z_critico

        # ── Métricas ──
        st.subheader("Resultados de la prueba Z")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Media muestral (x̄)", f"{x_bar:.4f}")
        r2.metric("Estadístico Z",       f"{Z_calc:.4f}")
        r3.metric("p-value",             f"{p_value:.4f}")
        r4.metric("Z crítico",           f"{Z_critico:.4f}")

        # ── Gráfica región crítica ──
        st.subheader("Gráfica de la Región Crítica")

        x_plot = np.linspace(-4, 4, 1000)
        y_plot = stats.norm.pdf(x_plot, 0, 1)

        fig_z, ax_z = plt.subplots(figsize=(10, 4))
        ax_z.plot(x_plot, y_plot, color="black", label="Normal Estándar")

        if "Bilateral" in tipo_prueba:
            ax_z.fill_between(x_plot, 0, y_plot,
                              where=(x_plot > Z_critico) | (x_plot < -Z_critico),
                              color="red", alpha=0.3, label="Zona de Rechazo")
        elif "izquierda" in tipo_prueba:
            ax_z.fill_between(x_plot, 0, y_plot,
                              where=(x_plot < Z_critico),
                              color="red", alpha=0.3, label="Zona de Rechazo")
        else:
            ax_z.fill_between(x_plot, 0, y_plot,
                              where=(x_plot > Z_critico),
                              color="red", alpha=0.3, label="Zona de Rechazo")

        ax_z.axvline(Z_calc, color="blue", linestyle="--", linewidth=2,
                     label=f"Z calculado ({Z_calc:.2f})")
        ax_z.set_title("Visualización de la Decisión Estadística")
        ax_z.set_xlabel("Desviaciones Estándar (Z)")
        ax_z.legend()
        st.pyplot(fig_z)

        # ── Decisión ──
        if rechazar:
            st.error(f"🔴 Se RECHAZA H₀ (p = {p_value:.4f} ≤ α = {alpha})")
        else:
            st.success(f"🟢 No se rechaza H₀ (p = {p_value:.4f} > α = {alpha})")

        # ─────────────────────────────────────────────
        #  MÓDULO 4 – ASISTENTE DE IA (GEMINI)
        # ─────────────────────────────────────────────
        st.divider()
        st.header("4. Asistente de IA – Interpretación de Resultados")

        if not gemini_listo:
            st.warning("⚠️ Configura tu ANTHROPIC_API_KEY en ia_helper.py para activar la IA.")
        else:
            prompt_ia = f"""Actúa como un experto en estadística. Se realizó una prueba Z con estos datos:
- Variable: {columna}
- Media muestral: {x_bar:.4f}
- Media hipotética (H0): {mu0:.2f}
- Desviación estándar poblacional: {sigma_pob:.4f}
- Tamaño de muestra (n): {n}
- Nivel de significancia (α): {alpha}
- Tipo de prueba: {tipo_prueba}
- Estadístico Z calculado: {Z_calc:.4f}
- p-value obtenido: {p_value:.4f}
- Decisión: {"Se RECHAZA H₀" if rechazar else "No se rechaza H₀"}
- Sesgo de los datos (skewness): {skewness:.4f}
- Curtosis: {kurtosis:.4f}

Explica brevemente qué significa este resultado y si el sesgo afecta la validez de la prueba."""

            with st.expander("Ver prompt enviado a Gemini"):
                st.code(prompt_ia, language="text")

            # El botón SOLO llama a la API y guarda en session_state
            if st.button("🤖 Consultar con el Asistente de IA"):
                with st.spinner("La IA está analizando tus resultados..."):
                    respuesta, error = consultar_ia(gemini_model, prompt_ia)
                    st.session_state.respuesta_ia = respuesta
                    st.session_state.error_ia     = error

            # Renderizar respuesta FUERA del botón (evita el error de React/DOM)
            if st.session_state.error_ia:
                st.error(st.session_state.error_ia)

            if st.session_state.respuesta_ia:
                st.markdown("### 🤖 Interpretación del Asistente:")
                st.info(st.session_state.respuesta_ia)