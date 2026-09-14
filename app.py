
import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

modelo = joblib.load(
    os.path.join(BASE_DIR, "modelo_knn_iris.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler_iris.pkl")
)

variables = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

st.set_page_config(
    page_title="Clasificación Iris con K-NN",
    page_icon="🌸"
)

st.title("🌸 Clasificación de agrupaciones - Iris")

st.write(
    "Ingrese las características de una nueva flor para "
    "determinar a qué agrupación pertenece."
)

sepal_length = st.number_input(
    "Longitud del sépalo (cm)",
    min_value=0.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Ancho del sépalo (cm)",
    min_value=0.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Longitud del pétalo (cm)",
    min_value=0.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Ancho del pétalo (cm)",
    min_value=0.0,
    value=0.2,
    step=0.1
)

if st.button("Clasificar"):

    nueva_flor = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=variables
    )

    nueva_flor_scaled = scaler.transform(nueva_flor)

    cluster = modelo.predict(
        nueva_flor_scaled
    )[0]

    probabilidad = modelo.predict_proba(
        nueva_flor_scaled
    )[0].max() * 100

    st.success(
        f"La observación pertenece al Cluster {cluster}"
    )

    st.write(
        f"Coincidencia con la agrupación: {probabilidad:.2f}%"
    )
