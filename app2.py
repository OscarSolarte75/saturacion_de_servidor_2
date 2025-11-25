import streamlit as st
import numpy as np
import pandas as pd

# ---------------------------------------------------------
#   RESOLVER p(t) SIN SCIPY (Método de Newton)
# ---------------------------------------------------------

def resolver_p(t, C, max_iter=50, tol=1e-10):
    """
    Resuelve la ecuación implícita:
        t^2 * p + t * exp(p) = C
    usando el método de Newton-Raphson sin scipy.
    """

    # Estimación inicial adaptativa
    p = 0.1 if t < 2 else -0.1

    for _ in range(max_iter):
        F = t**2 * p + t * np.exp(p) - C
        F_prime = t**2 + t * np.exp(p)

        # Actualización Newton
        p_new = p - F / F_prime

        if abs(p_new - p) < tol:
            return p_new

        p = p_new

    return p  # devuelve lo que tenga aunque no haya convergido del todo


def derivada_p(t, p):
    """
    p'(t) = -(2 t p + e^p) / (t^2 + t e^p)
    """
    num = -(2 * t * p + np.exp(p))
    den = (t**2 + t * np.exp(p))
    return num / den


# ---------------------------------------------------------
#   INTERFAZ STREAMLIT
# ---------------------------------------------------------

st.title("Modelo Implícito del Servidor Web")

st.sidebar.header("Controles del Modelo Matemático")

# Constante de la ecuación implícita
C = st.sidebar.slider(
    "Constante C de t² p + t e^p = C",
    0.5, 5.0, 2.1487, 0.0001
)

t_max = st.sidebar.slider(
    "Tiempo máximo de simulación",
    1, 50, 10
)

res = st.sidebar.slider(
    "Resolución de muestreo",
    50, 2000, 300, 50
)

# -----------------------------------------
# Calcular valores
# -----------------------------------------

t_values = np.linspace(1, t_max, res)
p_vals = np.array([resolver_p(t, C) for t in t_values])
dp_vals = np.array([derivada_p(t, p_vals[i]) for i, t in enumerate(t_values)])

df_graph = pd.DataFrame({
    "t": t_values,
    "p(t)": p_vals,
    "p'(t)": dp_vals
}).set_index("t")

# -----------------------------------------
# GRÁFICAS
# -----------------------------------------

st.subheader("Evolución de p(t)")
st.line_chart(df_graph["p(t)"])

st.subheader("Evolución de p'(t)")
st.line_chart(df_graph["p'(t)"])

# -----------------------------------------
# TABLA
# -----------------------------------------

st.subheader("Tabla de valores (cada 0.5)")

t_table = np.arange(1.0, t_max + 0.001, 0.5)
p_table = np.array([resolver_p(t, C) for t in t_table])
dp_table = np.array([derivada_p(t, p_t) for t, p_t in zip(t_table, p_table)])

df_table = pd.DataFrame({
    "t": np.round(t_table, 3),
    "p(t)": np.round(p_table, 6),
    "p'(t)": np.round(dp_table, 6)
})

st.dataframe(df_table)
