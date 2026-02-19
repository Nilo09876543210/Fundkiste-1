import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Modell laden
# -----------------------------
@st.cache_resource
def load_model():
    # compile=False vermeidet Fehler bei manchen TM-Modellen
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

# Klassen (müssen genau der Reihenfolge in Teachable Machine entsprechen!)
class_names = ["blau", "gelb", "rot", "keine davon"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("👕 T-Shirt Farb-Erkennung")
st.write("Lade ein Bild hoch, und das Modell erkennt die Farbe des T-Shirts.")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Bild öffnen und in RGB konvertieren
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten: Resize & Normalisierung
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension

    # -----------------------------
    # Vorhersage
    # -----------------------------
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[0][predicted_index])

    st.markdown("### ✅ Ergebnis")
    st.write(f"**Farbe:** {class_names[predicted_index]}")
    st.write(f"**Sicherheit:** {confidence * 100:.1f}%")

    if confidence < 0.6:
        st.warning("⚠️ Geringe Sicherheit – das Ergebnis könnte ungenau sein.")

    # Optional: alle Wahrscheinlichkeiten anzeigen
    st.markdown("### 📊 Wahrscheinlichkeiten pro Farbe")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {predictions[0][i]*100:.1f}%")
