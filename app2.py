import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------
#   MODELO MATEMÁTICO
# -----------------------------

def carga_p(t, c1, c2):
    return (c1 / t**2) - c2 * t

def derivada_p(t, c1, c2):
    return (-2 * c1) / t**3 - c2


# -----------------------------
#   PARÁMETROS
# -----------------------------
c1 = 2.1487
c2 = 0.5

st.title("📈 Modelo de Carga de un Servidor Web")

st.sidebar.header("Parámetros del modelo")
st.sidebar.write(f"**c1 =** {c1}")
st.sidebar.write(f"**c2 =** {c2}")

t_max = st.sidebar.slider("Tiempo máximo (t)", 5, 50, 25)
t_values = np.linspace(1, t_max, 600)

# Cálculos
p_vals = carga_p(t_values, c1, c2)
dp_vals = derivada_p(t_values, c1, c2)

# Convertir todo en DataFrame para graficar
df_graph = pd.DataFrame({
    "t": t_values,
    "Carga p(t)": p_vals,
    "Derivada p'(t)": dp_vals
}).set_index("t")

# -----------------------------
#   GRÁFICA p(t)
# -----------------------------
st.subheader("Evolución de la carga p(t)")
st.line_chart(df_graph["Carga p(t)"])

# -----------------------------
#   GRÁFICA p'(t)
# -----------------------------
st.subheader("Tasa de cambio p'(t)")
st.line_chart(df_graph["Derivada p'(t)"])

# -----------------------------
#   TABLA DE VALORES
# -----------------------------
st.subheader("Tabla de valores p(t) y p'(t)")

t_table = np.arange(1, min(t_max, 25)+1, 2)
df_table = pd.DataFrame({
    "Tiempo t": t_table,
    "Carga p(t)": np.round(carga_p(t_table, c1, c2), 4),
    "Velocidad p'(t)": np.round(derivada_p(t_table, c1, c2), 6)
})

st.dataframe(df_table)
