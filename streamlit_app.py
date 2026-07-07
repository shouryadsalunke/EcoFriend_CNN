streamlit_code = """
import os
import json
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Config & Paths
MODEL_PATH = "../models/best_scratch_model.keras"
METRICS_PATH = "metrics.json"
DASHBOARD_PATH = "evaluation_dashboard.png"
CURVE_PATH = "accuracy_curve.png"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.set_page_config(page_title="EcoFriend-CNN Smart Bin", layout="wide")

# Load model with caching so it doesn't reload on every button click
@st.cache_resource
def load_waste_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_waste_model()

# Title banner
st.title("EcoFriend-CNN Smart Sorting Interface")
st.write("An Edge-Optimized Waste Classification Pipeline")

# Create the required tabs from your project brief
tab1, tab2 = st.tabs(["Tab 1 (Smart Sorting)", "Tab 2 (Model Analytics)"])

# ==========================================
# TAB 1: SMART SORTING & WEBCAM INTERFACE
# ==========================================
with tab1:
    st.header("Real-Time Waste Classification")
    
    # Selection widget allowing webcam or manual upload
    input_method = st.radio("Select Input Source:", ("Webcam Input", "Upload Image File"))
    
    input_image = None
    
    if input_method == "Webcam Input":
        # Streamlit's native browser webcam component
        camera_file = st.camera_input("Hold the trash item up to your webcam camera")
        if camera_file:
            input_image = Image.open(camera_file)
    else:
        uploaded_file = st.file_uploader("Upload an image asset...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Target Image Source", width=350)

    # Execution Block
    if input_image is not None:
        if st.button("Trigger Trash Identification"):
            if model is None:
                st.error("Model file not found! Please check your file paths.")
            else:
                with st.spinner("Processing visual features..."):
                    # Match your exact pipeline image preprocessing requirements
                    img = input_image.resize((128, 128))
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0) # Create batch axis
                    
                    # Predict
                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    predicted_class = CLASSES[np.argmax(predictions[0])]
                    confidence = 100 * np.max(predictions[0])
                    
                    # Output Results
                    st.success(f"**Analysis Result:** Item identified as **{predicted_class.upper()}** ({confidence:.2f}% confidence)")
                    
                    # Dummy instructions mapping based on class selection
                    instructions = {
                        "cardboard": "Place in blue recycling bin. Flatten boxes to save space.",
                        "glass": "Rinse out food residue and place in glass recycling receptacle.",
                        "metal": "Clean thoroughly. Acceptable in metal/aluminum recycling points.",
                        "paper": "Recycle in dry paper bins. Ensure no oil staining present.",
                        "plastic": "Verify recycling codes on container bottom. Place in plastic sorting tray.",
                        "trash": "Non-recyclable material. Route directly to general landfill bin."
                    }
                    st.info(f"**Disposal Protocol:** {instructions.get(predicted_class, 'Route to standard waste sorting bin.')}")

# ==========================================
# TAB 2: MODEL PERFORMANCE ANALYTICS
# ==========================================
with tab2:
    st.header("Pipeline Evaluation & Quantization Insights")
    
    # Read metrics.json automatically generated from training script
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            data = json.load(f)
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Floating-Point Model")
            st.metric("Test Accuracy", f"{data['raw_model']['test_accuracy']*100:.2f}%")
            st.text(f"File Size: {data['raw_model']['file_size_mb']} MB")
            st.text(f"Latency per Image: {data['raw_model']['average_latency_ms']} ms")
            
        with col2:
            st.subheader("Quantized INT8 Model (PTQ)")
            st.metric("Quantized Accuracy", f"{data['quantized_model']['test_accuracy']*100:.2f}%")
            st.text(f"File Size: {data['quantized_model']['file_size_mb']} MB")
            st.text(f"Latency per Image: {data['quantized_model']['average_latency_ms']} ms")
    else:
        st.warning("`metrics.json` file missing. Complete the main model execution pipeline first.")
        
    # Display the pre-rendered analytics plots side-by-side
    st.subheader("Visual Analytics Performance Dashboard")
    if os.path.exists(DASHBOARD_PATH):
        st.image(DASHBOARD_PATH, use_container_width=True)
    if os.path.exists(CURVE_PATH):
        st.image(CURVE_PATH, width=800)
"""

# Changed the output file name to 'streamlit_app.py' to prevent conflicts with your backend app.py
with open('/kaggle/working/app/streamlit_app.py', 'w') as f:
    f.write(streamlit_code.strip())
print("Streamlit web app engine script written successfully to /kaggle/working/app/streamlit_app.py")
