import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import glob
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from model import get_model  # Import our new factory

# Config
IMG_SIZE = 224
SEQ_LEN = 16  # Reduced slightly to fit Transformers in memory
BATCH_SIZE = 4


class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.video_paths = []
        self.labels = []
        norm_files = glob.glob(os.path.join(root_dir, 'normal', '*.mp4'))
        abnorm_files = glob.glob(os.path.join(root_dir, 'abnormal', '*.mp4'))
        self.video_paths.extend(norm_files + abnorm_files)
        self.labels.extend([0] * len(norm_files) + [1] * len(abnorm_files))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < SEQ_LEN:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ensure RGB
                frame = frame.astype(np.float32) / 255.0
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)
            else:
                frames.append(np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32))
        cap.release()
        return torch.tensor(np.array(frames[:SEQ_LEN])), torch.tensor(label)


def run_training(model_name, epochs=5, status_placeholder=None):
    """
    Main training function callable by Streamlit.
    status_placeholder: A Streamlit element to write logs to.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if status_placeholder:
        status_placeholder.text(f"Initializing {model_name} on {device}...")

    # Load Data
    if not os.path.exists('dataset'):
        return "Error: Dataset not found."

    full_dataset = VideoDataset(root_dir='dataset')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Specific Model
    model = get_model(model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        msg = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}"
        print(msg)
        if status_placeholder:
            status_placeholder.text(msg)

    # Save with Unique Name
    # e.g., "model_3D_CNN.pth"
    safe_name = model_name.replace(" ", "_")
    save_path = f"model_{safe_name}.pth"
    torch.save(model.state_dict(), save_path)

    return f"Success! Model saved as {save_path}"


if __name__ == "__main__":
    # Default behavior if run directly
    run_training("CNN-LSTM")