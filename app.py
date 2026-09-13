
import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y escalador
modelo = joblib.load("modelo_knn.pkl")
scaler = joblib.load("scaler_kmeans.pkl")

# Variables utilizadas durante el entrenamiento
variables = [
    "Customer_care_calls",
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Discount_offered",
    "Weight_in_gms"
]

# Configuración de la página
st.set_page_config(
    page_title="Clasificación logística de envíos",
    page_icon="📦"
)

st.title("📦 Clasificación logística de envíos")

st.write(
    "Ingrese las características de un nuevo envío para "
    "determinar el segmento logístico al que pertenece."
)

# Entradas del usuario
customer_care_calls = st.number_input(
    "Número de llamadas a servicio al cliente",
    min_value=0,
    step=1
)

customer_rating = st.slider(
    "Calificación del cliente",
    min_value=1,
    max_value=5,
    value=3
)

cost_product = st.number_input(
    "Costo del producto",
    min_value=0.0,
    step=1.0
)

prior_purchases = st.number_input(
    "Número de compras anteriores",
    min_value=0,
    step=1
)

discount = st.number_input(
    "Descuento ofrecido",
    min_value=0.0,
    step=1.0
)

weight = st.number_input(
    "Peso del producto en gramos",
    min_value=0.0,
    step=10.0
)

# Botón de clasificación
if st.button("Clasificar envío"):

    nuevo_envio = pd.DataFrame(
        [[
            customer_care_calls,
            customer_rating,
            cost_product,
            prior_purchases,
            discount,
            weight
        ]],
        columns=variables
    )

    # Aplicar el mismo escalador usado en el entrenamiento
    nuevo_envio_scaled = scaler.transform(
        nuevo_envio
    )

    # Clasificar mediante K-NN
    cluster = modelo.predict(
        nuevo_envio_scaled
    )[0]

    # Proporción de vecinos del cluster seleccionado
    probabilidades = modelo.predict_proba(
        nuevo_envio_scaled
    )[0]

    confianza = probabilidades.max() * 100

    st.success(
        f"El nuevo envío pertenece al Cluster {cluster}"
    )

    st.write(
        f"Coincidencia con el segmento: {confianza:.2f}%"
    )

    st.subheader("Información del envío")
    st.dataframe(nuevo_envio)

    st.info(
        "La clasificación permite asignar el nuevo envío "
        "a uno de los segmentos logísticos identificados "
        "previamente mediante K-Means."
    )
