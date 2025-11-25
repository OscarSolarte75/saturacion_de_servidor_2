import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Modelo de carga p(t)
def carga_p(t, c1, c2):
    return (c1 / t ** 2) - c2 * t

# Derivada de p(t)
def derivada_p(t, c1, c2):
    return (-2 * c1) / (t ** 3) - c2

# Constantes del modelo
c1 = 2.1487
c2 = 0.5

st.title("Modelo de Carga de Servidor Web")

st.sidebar.header("Parámetros fijos")
st.sidebar.write(f"c1 = {c1}, c2 = {c2}")

# Tiempo máximo para visualización
t_max = st.sidebar.slider("Tiempo máximo t", 5, 50, 25)
t_values = np.linspace(1, t_max, 600)

# Cálculo de valores
p_vals = carga_p(t_values, c1, c2)
dp_vals = derivada_p(t_values, c1, c2)

# Gráfica de carga p(t)
fig1, ax1 = plt.subplots()
ax1.plot(t_values, p_vals, color='blue')
ax1.set_xlabel("Tiempo t")
ax1.set_ylabel("Carga p(t)")
ax1.set_title("Evolución de la carga del servidor")
ax1.grid(True)
st.pyplot(fig1)

# Gráfica de derivada p'(t)
fig2, ax2 = plt.subplots()
ax2.plot(t_values, dp_vals, color='green')
ax2.axhline(0, color='black', linestyle='--')
ax2.set_xlabel("Tiempo t")
ax2.set_ylabel("Derivada p'(t)")
ax2.set_title("Tasa de cambio de la carga")
ax2.grid(True)
st.pyplot(fig2)

# Tabla de valores
t_table = np.arange(1, min(t_max, 25)+1, 2)
p_table = carga_p(t_table, c1, c2)
dp_table = derivada_p(t_table, c1, c2)
df = pd.DataFrame({
    'Tiempo t': t_table,
    'Carga p(t)': np.round(p_table, 4),
    "Velocidad p'(t)": np.round(dp_table, 6)
})
st.write("### Valores de carga y su cambio")
st.dataframe(df)