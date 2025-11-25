import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# ---------------------------------------------------------
#   ECUACIÓN IMPLÍCITA DEL MODELO
#   t^2 p(t) + t e^p(t) = C
# ---------------------------------------------------------

def resolver_p(t, C):
    """
    Resuelve p(t) numéricamente de la ecuación implícita:
        t^2 * p + t * e^p = C
    usando fsolve con estimación inicial adaptativa.
    """
    # Estimación inicial razonable
    p0 = -0.1 if t > 2 else 0.5
    eq = lambda p: t**2 * p + t * np.exp(p) - C
    return fsolve(eq, p0)[0]


def derivada_p(t, p):
    """
    p'(t) = -(2tp + e^p) / (t^2 + t e^p)
    """
    return -(2 * t * p + np.exp(p)) / (t**2 + t * np.exp(p))

# ---------------------------------------------------------
#   INTERFAZ
# ---------------------------------------------------------

st.title("Modelo de Ecuación Implícita del Servidor Web")

st.sidebar.header("Controles del Modelo Matemático")

# Constante C de la ecuación implícita
C = st.sidebar.slider(
    "Constante C de la ecuación t² p + t e^p = C",
    0.5, 5.0, 2.1487, 0.0001
)

t_max = st.sidebar.slider(
    "Tiempo máximo de simulación (t)",
    1, 50, 10
)

resolucion = st.sidebar.slider(
    "Resolución de muestreo (n puntos)",
    50, 2000, 400, 50
)

# Dominio del tiempo
t_values = np.linspace(1, t_max, resolucion)

# Resolver p(t) para cada t
p_vals = np.array([resolver_p(t, C) for t in t_values])

# Calcular derivada p'(t)
dp_vals = np.array([derivada_p(t, p_vals[i]) for i, t in enumerate(t_values)])

# DataFrame para gráficas
df_graph = pd.DataFrame({
    "t": t_values,
    "Carga p(t)": p_vals,
    "Derivada p'(t)": dp_vals
}).set_index("t")

# ---------------------------------------------------------
#   GRÁFICAS
# ---------------------------------------------------------
st.subheader("Evolución de p(t)")
st.line_chart(df_graph["Carga p(t)"])

st.subheader("Evolución de la derivada p'(t)")
st.line_chart(df_graph["Derivada p'(t)"])

# ---------------------------------------------------------
#   TABLA
# ---------------------------------------------------------
st.subheader("Tabla de Valores (cada 0.5 unidades)")

t_table = np.arange(1.0, min(t_max, 50) + 0.1, 0.5)
p_table = [resolver_p(t, C) for t in t_table]
dp_table = [derivada_p(t, p_table[i]) for i, t in enumerate(t_table)]

df_table = pd.DataFrame({
    "t": np.round(t_table, 3),
    "p(t)": np.round(p_table, 6),
    "p'(t)": np.round(dp_table, 6)
})

st.dataframe(df_table)
