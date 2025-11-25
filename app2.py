import streamlit as st
import numpy as np
import pandas as pd

# ---------------------------------------------------------
#   MODELO MATEMÁTICO
# ---------------------------------------------------------

def carga_p(t, c1, c2):
    return (c1 / t**2) - c2 * t

def derivada_p(t, c1, c2):
    return (-2 * c1) / t**3 - c2


# ---------------------------------------------------------
#   INTERFAZ
# ---------------------------------------------------------

st.title("Modelo de Carga de un Servidor Web")

st.sidebar.header("Controles del Modelo Matemático")

# Sliders con nombres formales
c1 = st.sidebar.slider(
    "Constante de crecimiento (c1)",
    0.1, 10.0, 2.1487, 0.1
)

c2 = st.sidebar.slider(
    "Constante de pérdida temporal (c2)",
    0.1, 5.0, 0.5, 0.1
)

t_max = st.sidebar.slider(
    "Tiempo máximo de simulación (t)",
    5, 100, 25
)

resolucion = st.sidebar.slider(
    "Resolución de muestreo (n puntos)",
    100, 2000, 600, 100
)

# Dominio del tiempo
t_values = np.linspace(1, t_max, resolucion)

# Cálculos
p_vals = carga_p(t_values, c1, c2)
dp_vals = derivada_p(t_values, c1, c2)

# DataFrame para gráficas
df_graph = pd.DataFrame({
    "t": t_values,
    "Carga del servidor p(t)": p_vals,
    "Derivada de la carga p'(t)": dp_vals
}).set_index("t")


# ---------------------------------------------------------
#   GRÁFICA p(t)
# ---------------------------------------------------------
st.subheader("Evolución de la Carga del Servidor (p(t))")
st.line_chart(df_graph["Carga del servidor p(t)"])


# ---------------------------------------------------------
#   GRÁFICA p'(t)
# ---------------------------------------------------------
st.subheader("Tasa de Cambio de la Carga (p'(t))")
st.line_chart(df_graph["Derivada de la carga p'(t)"])


# ---------------------------------------------------------
#   TABLA DE VALORES
# ---------------------------------------------------------
st.subheader("Tabla de Valores de p(t) y p'(t)")

t_table = np.arange(1, min(t_max, 50) + 1, 2)

df_table = pd.DataFrame({
    "Tiempo (t)": t_table,
    "Carga p(t)": np.round(carga_p(t_table, c1, c2), 4),
    "Derivada p'(t)": np.round(derivada_p(t_table, c1, c2), 6)
})

st.dataframe(df_table)
