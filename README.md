# ♻️ EcoFriend

EcoFriend is an AI-powered smart waste segregation system that classifies household waste into six categories using a custom Convolutional Neural Network (CNN). Users upload an image of waste to receive an instant prediction, confidence score, and recycling/disposal guidance. The project also documents the tradeoffs between the original trained model and its TensorFlow Lite quantized version, built with efficient edge deployment (e.g. Raspberry Pi) in mind.

---
<img width="838" height="352" alt="Screenshot 2026-07-03 at 6 23 18 PM" src="https://github.com/user-attachments/assets/bfcb93e6-9644-4786-b927-61d6800c9356" />

---

## Features

- 🧠 Custom CNN (VGG-style backbone + Squeeze-and-Excitation attention blocks) trained from scratch using TensorFlow/Keras
- 📷 Real-time image-based waste classification via a simple web interface
- ♻️ Disposal and recycling recommendations for each predicted category
- ⚡ Deployed using a TensorFlow Lite (TFLite) INT8-quantized model — ~11x smaller than the original, for fast and lightweight inference
- 📊 Model analytics tab showing accuracy, size, latency, and parameter count
- 🌐 Built entirely with Streamlit (single app, no separate backend server)

---

## Waste Categories

The model classifies waste into the following categories:

- 📦 Cardboard
- 🍾 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model Optimization | TensorFlow Lite (Post-Training INT8 Quantization) |
| App Framework | Streamlit |
| Data Processing | NumPy, Pillow |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |

---

## Project Structure

\`\`\`text
EcoFriend/
│
├── app.py                      # Streamlit app — UI + inference logic
├── requirements.txt
├── README.md
│
├── quantized_model.tflite      # Deployed model (INT8, ~730 KB)
├── metrics.json                # Accuracy / size / latency stats
├── full_dashboard.png          # Combined evaluation dashboard
└── accuracy_curve.png          # Training accuracy & loss curves
\`\`\`

> Note: The original full-precision model (`best_scratch_model.keras`, ~8.3 MB) was used during training and evaluation but is not required by the deployed app, which runs entirely on the quantized model.

---

## Installation

Clone the repository:

\`\`\`bash
git clone https://github.com/<your-username>/EcoFriend.git
cd EcoFriend
\`\`\`

Install the required dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## Running the Application

\`\`\`bash
streamlit run app.py
\`\`\`

The application will open in your default web browser.

---

## Model

- **Architecture:** Custom CNN — 4 convolutional blocks (32 → 64 → 128 → 256 filters) with Squeeze-and-Excitation attention blocks, batch normalization, and dropout
- **Input Size:** 128 × 128 RGB images
- **Total Parameters:** ~675K (~2.6 MB in full precision)
- **Optimizer:** Adam with a warmup + cosine decay learning rate schedule
- **Loss Function:** Categorical Crossentropy with label smoothing
- **Training Techniques:**
  - Data augmentation (flip, rotation, zoom, translation, contrast, brightness)
  - Balanced class sampling (equal exposure per category regardless of dataset imbalance)
  - Early stopping with best-weight restoration
  - Post-training INT8 quantization for deployment

---

## Performance

| Metric | Original Model | Quantized Model |
|---|---|---|
| Test Accuracy | 97.37% | 96.84% |
| File Size | 7.92 MB | 0.70 MB |
| Parameters | 675,206 | 675,206 |

Full metrics are available in `metrics.json` and visualized in `full_dashboard.png`, including a confusion matrix, per-class F1 scores, and size/latency/accuracy comparisons.

> Note: raw benchmark latency numbers can be misleading — the quantized model was timed via single-image interpreter calls, while the original model was timed via batched prediction, so the two aren't directly comparable as measured. TFLite's INT8 quantization is primarily optimized for ARM-based edge devices (e.g. Raspberry Pi), where it's expected to perform well.

---

## Future Improvements

- 📸 Webcam-based real-time waste detection
- 🥧 Raspberry Pi deployment using the quantized TFLite model
- 📱 Mobile-friendly interface
- ♻️ Support for additional waste categories
- 📍 Nearby recycling center recommendations

---

## Author

**Shourya Salunke**

---

## License

This project is intended for educational purposes.
