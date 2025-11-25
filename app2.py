import streamlit as st
import numpy as np
import pandas as pd

def resolver_p(t, C, it=40, tol=1e-10):
    p = 0.1 if t < 2 else -0.1
    for _ in range(it):
        F = t**2 * p + t*np.exp(p) - C
        Fp = t**2 + t*np.exp(p)
        p_new = p - F/Fp
        if abs(p_new - p) < tol:
            break
        p = p_new
    return p

def derivada_p(t, p):
    return -(2*t*p + np.exp(p)) / (t**2 + t*np.exp(p))

st.title("Modelo Implícito del Servidor Web")

st.sidebar.header("Punto Inicial")

# ------- t0 -------
t0 = st.sidebar.slider("t₀", 1.0, 10.0, 1.0, 0.5, key="t0_slider")
t0 = st.sidebar.number_input("Editar t₀", 1.0, 10.0, t0, 0.1, key="t0_input")

# ------- p0 -------
p0 = st.sidebar.slider("p(t₀)", -1.0, 2.0, 0.5, 0.1, key="p0_slider")
p0 = st.sidebar.number_input("Editar p(t₀)", -1.0, 2.0, p0, 0.1, key="p0_input")

C = t0**2 * p0 + t0*np.exp(p0)

t_vals = np.linspace(1, 20, 400)
p_vals = np.array([resolver_p(t, C) for t in t_vals])
dp_vals = np.array([derivada_p(t, p) for t, p in zip(t_vals, p_vals)])

df = pd.DataFrame({"t": t_vals, "p(t)": p_vals, "p'(t)": dp_vals}).set_index("t")

st.subheader("p(t)")
st.line_chart(df["p(t)"])

st.subheader("p'(t)")
st.line_chart(df["p'(t)"])

t_tab = np.arange(1, 21, 1)
p_tab = np.array([resolver_p(t, C) for t in t_tab])
dp_tab = np.array([derivada_p(t, p) for t, p in zip(t_tab, p_tab)])

st.subheader("Tabla")
st.dataframe(pd.DataFrame({"t": t_tab, "p(t)": p_tab, "p'(t)": dp_tab}))
