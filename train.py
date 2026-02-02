import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import glob
import os
import numpy as np
from model import get_model
from metrics_manager import MetricsManager  # <--- IMPORT NEW CLASS

# --- CONFIG ---
IMG_SIZE = 224
SEQ_LEN = 16
BATCH_SIZE = 8


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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)
            else:
                frames.append(np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32))
        cap.release()
        return torch.tensor(np.array(frames[:SEQ_LEN])), torch.tensor(label)


def run_training(model_name, epochs=5, status_placeholder=None):
    # 1. Initialize Metrics Manager
    metrics_manager = MetricsManager(model_name)

    # 2. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        msg = f"✅ Training on GPU: {torch.cuda.get_device_name(0)}"
    else:
        device = torch.device("cpu")
        msg = "⚠️ Training on CPU."

    print(msg)
    if status_placeholder:
        status_placeholder.text(msg)

    # 3. Load Data
    if not os.path.exists('dataset'):
        return "Error: Dataset not found. Run preprocess.py first."

    full_dataset = VideoDataset(root_dir='dataset')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Validation Loader

    # 4. Initialize Model
    model = get_model(model_name).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- START TIMER ---
    metrics_manager.start_training_timer()

    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        log = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}"
        print(log)
        if status_placeholder:
            status_placeholder.text(log)

    # --- STOP TIMER ---
    total_time = metrics_manager.stop_training_timer()

    # 6. Save Model
    safe_name = model_name.replace(" ", "_")
    save_path = f"model_{safe_name}.pth"
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)

    # 7. Final Evaluation & Metrics Saving
    if status_placeholder:
        status_placeholder.text("Running final evaluation...")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate & Save
    stats = metrics_manager.compute_metrics(all_labels, all_preds, all_probs)
    metrics_manager.save_metrics(stats, total_time)

    # Generate Plots
    metrics_manager.plot_confusion_matrix(np.array(stats['confusion_matrix']))
    metrics_manager.plot_roc_curve(stats['roc_data']['fpr'], stats['roc_data']['tpr'], stats['auc'])

    return f"Success! Model saved to {save_path}. Training took {total_time:.2f}s."


if __name__ == "__main__":
    run_training("CNN-LSTM")