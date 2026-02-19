import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Modell laden
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Klassen (Reihenfolge muss exakt deinem Training entsprechen!)
class_names = ["gelb", "blau", "rot", "keine davon"]

st.title("T-Shirt Farb-Erkennung")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bildgröße anpassen (WICHTIG: gleiche Größe wie im Training!)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))

    st.write("### Ergebnis:")
    st.write(f"Farbe: **{predicted_class}**")
    st.write(f"Sicherheit: {confidence:.2f}")
