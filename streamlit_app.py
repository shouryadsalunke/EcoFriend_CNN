import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Config 
IMG_SIZE = 128
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

DISPOSAL_GUIDANCE = {
    'cardboard': "♻️ Recyclable. Flatten boxes and keep dry before placing in the recycling bin.",
    'glass':     "♻️ Recyclable. Rinse out any residue. Check local rules for color-sorted glass.",
    'metal':     "♻️ Recyclable. Rinse cans/tins. Aluminum and steel are both widely accepted.",
    'paper':     "♻️ Recyclable. Keep clean and dry — greasy or food-stained paper should go to trash.",
    'plastic':   "♻️ Recyclable (check resin code). Rinse containers and remove caps if required locally.",
    'trash':     "🚮 Not recyclable. Dispose of in general waste.",
}

QUANT_MODEL_PATH = "quantized_model.tflite"
METRICS_PATH = "metrics.json"
DASHBOARD_IMAGE_PATH = "evaluation_dashboard.png"

st.set_page_config(page_title="EcoFriend", page_icon="♻️", layout="centered")

@st.cache_resource
def load_tflite_interpreter():
    interpreter = tf.lite.Interpreter(model_path=QUANT_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict_tflite(interpreter, img_array):
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    interpreter.set_tensor(in_det[0]["index"], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(out_det[0]["index"])[0]

st.title("♻️ EcoFriend")
st.caption("AI-powered smart waste segregation — snap a photo or upload an image to classify it into one of six categories.")

tab_classify, tab_analytics = st.tabs(["🔍 Classify Waste", "📊 Model Analytics"])

with tab_classify:
    # Radio toggle to pick camera input vs upload
    input_method = st.radio("Select Input Source:", ("Webcam Input", "Upload Image File"))
    
    image = None
    
    if input_method == "Webcam Input":
        # Streamlit's native browser webcam component
        camera_file = st.camera_input("Hold the trash item up to your webcam camera")
        if camera_file:
            image = Image.open(camera_file)
    else:
        uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)

    if image is not None:
        img_array = preprocess_image(image)

        with st.spinner("Classifying..."):
            interpreter = load_tflite_interpreter()
            preds = predict_tflite(interpreter, img_array)

        top_idx = int(np.argmax(preds))
        top_class = CLASSES[top_idx]
        confidence = float(preds[top_idx]) * 100

        st.subheader(f"Prediction: **{top_class.capitalize()}**")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.info(DISPOSAL_GUIDANCE[top_class])

        st.write("Confidence by category:")
        st.bar_chart({cls: float(p) for cls, p in zip(CLASSES, preds)})

with tab_analytics:
    st.subheader("Quantized Model Performance")

    metrics = load_metrics()
    if metrics:
        quant = metrics["quantized_model"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test accuracy", f"{quant['test_accuracy']*100:.2f}%")
        col2.metric("File size", f"{quant['file_size_mb']} MB")
        col3.metric("Avg latency", f"{quant['average_latency_ms']} ms")
        col4.metric("Parameters", f"{quant['trainable_parameters']:,}")
    else:
        st.warning("metrics.json not found — analytics unavailable.")

    st.subheader("Evaluation Dashboard")
    try:
        # Fixed parameter from use_column_width to use_container_width to clear warning
        st.image(DASHBOARD_IMAGE_PATH, use_container_width=True)
    except Exception:
        st.warning(f"{DASHBOARD_IMAGE_PATH} not found.")
