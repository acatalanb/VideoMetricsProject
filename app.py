"""
app.py - Video Classification Web Application

This is the main Streamlit application that provides a user-friendly interface for:
1. Dataset Selection: Choose between Kvasir Chest X-Ray and UCF Crime datasets
2. Model Training: Train CNN-LSTM, 3D CNN, or Video Transformer architectures
3. Live Analysis: Load trained models and run inference on uploaded videos

The app displays real-time performance metrics, confusion matrices, ROC curves,
inference times, and probability distributions for predictions. Supports both
CPU and GPU execution with automatic detection.

Author: Video Metrics Project Team
Created: 2026-03-18
Version: 1.0.0-alpha
"""

import streamlit as st
import torch
import cv2
import numpy as np
import os
import time
import tempfile
import altair as alt
import pandas as pd
from model import get_model
from train import run_training
from metrics_manager import MetricsManager

KVASIR_DATASET_PATH = 'K:\VideoDataset\kvasir'
UCF_DATASET_PATH = 'K:\VideoDataset\crime-ucf'

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Platform", layout="wide", page_icon="🧬")

st.sidebar.title("🧬 AI Control Center")

# Initialize session state for dataset choice
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = KVASIR_DATASET_PATH

# Check GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    st.sidebar.success(f"🟢 GPU Active: {torch.cuda.get_device_name(0)} ({gpu_count} GPU{'s' if gpu_count > 1 else ''} available)")
else:
    DEVICE = torch.device("cpu")
    st.sidebar.error("🔴 CPU Mode")

app_mode = st.sidebar.selectbox("Select Mode", ["Dataset", "Train Model", "Run Analysis" ])
IMG_SIZE = 224
SEQ_LEN = 16


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // SEQ_LEN)
    for i in range(SEQ_LEN):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)
        else:
            frames.append(np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32))
    cap.release()
    return torch.tensor(np.array(frames)).unsqueeze(0).to(DEVICE)

# ==========================================
# MODE 1: TRAIN MODEL
# ==========================================
if app_mode == "Train Model":
    st.title("🛠️ Train a New Model")

    model_choice = st.selectbox("Choose Architecture", ["CNN-LSTM", "3D CNN", "Video Transformer"])
    epochs = st.slider("Epochs", 1, 20, 5)

    if st.button(f"Start Training {model_choice}"):
        dataset_path = st.session_state.dataset_path
        if not os.path.exists(dataset_path):
            st.error(f"No dataset found at {dataset_path}!")
        else:
            status_text = st.empty()
            with st.spinner("Training in progress..."):
                result = run_training(model_choice, epochs, status_text, dataset_path)
            st.success(result)

# ==========================================
# MODE 2: RUN ANALYSIS
# ==========================================
elif app_mode == "Run Analysis":
    st.title("🏥 Live Anomaly Detection")

    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    available_models = [f for f in os.listdir(cache_dir) if f.endswith('.pth') and f.startswith('model_')]

    if not available_models:
        st.warning("No models found. Please Train one first.")
    else:
        # --- MODEL SELECTION ---
        col_sel, col_stats = st.columns([1, 2])

        with col_sel:
            selected_file = st.selectbox("Select Trained Model", available_models)

            # Determine Architecture
            if "CNN-LSTM" in selected_file:
                arch = "CNN-LSTM"
            elif "3D_CNN" in selected_file:
                arch = "3D CNN"
            elif "Transformer" in selected_file:
                arch = "Video Transformer"
            else:
                arch = "CNN-LSTM"  # Fallback

            # Load Weights
            try:
                model = get_model(arch)
                model_path = os.path.join(cache_dir, selected_file)
                state_dict = torch.load(model_path, map_location=DEVICE)

                # Clean DataParallel keys if needed
                if list(state_dict.keys())[0].startswith('module.'):
                    from collections import OrderedDict

                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict[k[7:]] = v
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)

                model.to(DEVICE)
                model.eval()
                st.success(f"Loaded: {arch}")
            except Exception as e:
                st.error(f"Load Error: {e}")
                model = None

        # --- MODEL REPORT CARD (METRICS) ---
        with col_stats:
            metrics_mgr = MetricsManager(arch)
            stats = metrics_mgr.load_metrics()

            if stats:
                st.markdown("### 📊 Model Performance Report")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{stats['accuracy']:.1%}")
                m2.metric("Sensitivity", f"{stats['sensitivity']:.1%}")
                m3.metric("Specificity", f"{stats['specificity']:.1%}" if stats['specificity'] is not None else "N/A")
                m4.metric("Train Time", f"{stats['training_time_seconds']:.1f}s")

                with st.expander("View Detailed Plots"):
                    p1, p2 = st.columns(2)
                    if os.path.exists(metrics_mgr.cm_file):
                        p1.image(metrics_mgr.cm_file, caption="Confusion Matrix")
                    if os.path.exists(metrics_mgr.roc_file):
                        p2.image(metrics_mgr.roc_file, caption="ROC Curve")
            else:
                st.info("No training metrics found for this model.")

        st.divider()

        # --- LIVE INFERENCE ---
        st.subheader("Run Prediction")
        uploaded_file = st.file_uploader("Upload Scan", type=["mp4", "avi"])

        if uploaded_file and model:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()

            video_path = tfile.name
            st.video(video_path)

            if st.button("Analyze Video"):
                with st.spinner("Processing..."):
                    try:
                        # 1. Preprocessing Time
                        t0 = time.time()
                        input_tensor = process_video(video_path)

                        # 2. Inference Time
                        t1 = time.time()
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            conf, pred = torch.max(probs, 1)
                        t2 = time.time()

                        inference_time = t2 - t1
                        total_process_time = t2 - t0

                        # Display Results
                        res = {'Normal': probs[0][0].item(), 'Abnormal': probs[0][1].item()}

                        c1, c2 = st.columns([1, 1])
                        with c1:
                            if pred.item() == 1:
                                st.error(f"🚨 ANOMALY DETECTED")
                            else:
                                st.success(f"✅ NORMAL SCAN")
                            st.caption(f"Confidence: {conf.item():.1%}")

                        with c2:
                            st.metric("Inference Time", f"{inference_time * 1000:.0f} ms")
                            st.caption(f"Total Process Time: {total_process_time:.2f}s")

                        # Chart
                        df_chart = pd.DataFrame({"Class": list(res.keys()), "Probability": list(res.values())})
                        chart = alt.Chart(df_chart).mark_bar().encode(
                            x='Class', y='Probability',
                            color=alt.Color('Class', scale=alt.Scale(domain=['Normal', 'Abnormal'],
                                                                     range=['#2ecc71', '#e74c3c']))
                        )
                        st.altair_chart(chart, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        if os.path.exists(video_path):
                            os.unlink(video_path)

# ==========================================
# MODE 3: DATASET
# ==========================================
elif app_mode == "Dataset":
    st.title("Dataset to train")

    dataset_choice = st.selectbox("Select Dataset", ["Kvasir Chest X-Ray", "UCF Crime"])

    # Update dataset path based on selection
    if dataset_choice == "Kvasir Chest X-Ray":
        st.session_state.dataset_path = KVASIR_DATASET_PATH
    elif dataset_choice == "UCF Crime":
        st.session_state.dataset_path = UCF_DATASET_PATH

    st.info(f"Selected dataset path: {st.session_state.dataset_path}")