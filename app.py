import streamlit as st
import torch
import cv2
import numpy as np
import os
import tempfile
import altair as alt
import pandas as pd
from model import get_model
from train import run_training

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical AI Platform", layout="wide", page_icon="🧬")

# --- SIDEBAR & DIAGNOSTICS ---
st.sidebar.title("🧬 AI Control Center")

# Check GPU Status
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"🟢 {gpu_count} GPUs Detected")
    st.sidebar.caption(f"Primary: {gpu_name}")
    DEVICE = torch.device("cuda")
else:
    st.sidebar.error("🔴 CPU Only Mode")
    st.sidebar.caption("PyTorch cannot see your GPUs.")
    DEVICE = torch.device("cpu")

app_mode = st.sidebar.selectbox("Select Mode", ["Run Analysis", "Train Model"])

# --- SHARED CONSTANTS ---
IMG_SIZE = 224
SEQ_LEN = 16


# --- HELPER: VIDEO PREPROCESSING ---
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

    model_choice = st.selectbox(
        "Choose Architecture",
        ["CNN-LSTM", "3D CNN", "Video Transformer"]
    )

    epochs = st.slider("Epochs", 1, 20, 5)

    if st.button(f"Start Training {model_choice}"):
        if not os.path.exists("dataset"):
            st.error("No dataset found! Run 'preprocess.py' first.")
        else:
            status_text = st.empty()
            with st.spinner(f"Training on {DEVICE}..."):
                result_msg = run_training(model_choice, epochs, status_text)
            st.success(result_msg)

# ==========================================
# MODE 2: RUN ANALYSIS
# ==========================================
elif app_mode == "Run Analysis":
    st.title("🏥 Live Anomaly Detection")

    available_models = [f for f in os.listdir('.') if f.endswith('.pth') and f.startswith('model_')]

    if not available_models:
        st.warning("No trained models found. Train one first!")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            selected_weight_file = st.selectbox("Select Model", available_models)

            if "CNN-LSTM" in selected_weight_file:
                arch = "CNN-LSTM"
            elif "3D_CNN" in selected_weight_file:
                arch = "3D CNN"
            elif "Transformer" in selected_weight_file:
                arch = "Video Transformer"
            else:
                arch = st.selectbox("Architecture Type", ["CNN-LSTM", "3D CNN", "Video Transformer"])

            try:
                model = get_model(arch)
                state_dict = torch.load(selected_weight_file, map_location=DEVICE)

                # Handle DataParallel keys if loaded on single device
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
                st.success(f"Loaded {arch}")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                model = None

        with col2:
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

            if uploaded_file and model:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()  # Fix Windows lock

                video_path = tfile.name
                st.video(video_path)

                if st.button("Analyze Video"):
                    with st.spinner("Analyzing..."):
                        try:
                            input_tensor = process_video(video_path)
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                probs = torch.nn.functional.softmax(outputs, dim=1)
                                conf, pred = torch.max(probs, 1)

                            res = {'Normal': probs[0][0].item(), 'Abnormal': probs[0][1].item()}

                            if pred.item() == 1:
                                st.error(f"🚨 ANOMALY DETECTED ({res['Abnormal']:.1%})")
                            else:
                                st.success(f"✅ NORMAL SCAN ({res['Normal']:.1%})")

                            df_chart = pd.DataFrame({"Class": list(res.keys()), "Probability": list(res.values())})
                            chart = alt.Chart(df_chart).mark_bar().encode(
                                x='Class', y='Probability',
                                color=alt.Color('Class', scale=alt.Scale(domain=['Normal', 'Abnormal'],
                                                                         range=['#2ecc71', '#e74c3c']))
                            )
                            st.altair_chart(chart, use_container_width=True)

                        except Exception as e:
                            st.error(f"Analysis Error: {e}")
                        finally:
                            if os.path.exists(video_path):
                                os.unlink(video_path)