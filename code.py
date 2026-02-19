import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Modell laden (Teachable Machine)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

# Klassen (GENAU so wie in Teachable Machine!)
class_names = ["blau", "gelb", "rot", "keine davon"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("👕 T-Shirt Farb-Erkennung")
st.write("Lade ein Bild hoch und das Modell erkennt die Farbe.")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # -----------------------------
    # Bild vorbereiten (Teachable Machine Standard)
    # -----------------------------
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Vorhersage
    # -----------------------------
    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    confidence = float(predictions[0][index])

    st.markdown("### ✅ Ergebnis")
    st.write(f"**Farbe:** {class_names[index]}")
    st.write(f"**Sicherheit:** {confidence * 100:.1f}%")

    if confidence < 0.6:
        st.warning("⚠️ Geringe Sicherheit – Ergebnis könnte ungenau sein.")
")
