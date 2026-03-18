"""
model.py - Video Classification Model Architectures

This module contains three different deep learning architectures for video classification:
1. CNN-LSTM: Combines ResNet18 feature extraction with LSTM temporal modeling
2. 3D CNN: Uses ResNet R(2+1)D for spatiotemporal feature learning
3. Video Transformer: Leverages VideoMAE pretrained transformer architecture

The module provides a factory function get_model() to instantiate any of these architectures.

Author: Video Metrics Project Team
Created: 2026-03-18
Version: 1.0.0-alpha
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# Try importing Transformers (Handle case if user hasn't installed it)
try:
    from transformers import VideoMAEForVideoClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ==========================================
# 1. CNN-LSTM (The Baseline)
# ==========================================
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=256, lstm_layers=2):
        super(CNNLSTM, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # Input: (Batch, Seq, C, H, W)
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        cnn_out = self.feature_extractor(c_in)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out[:, -1, :])


# ==========================================
# 2. 3D CNN (ResNet R(2+1)D)
# ==========================================
class ResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet3D, self).__init__()
        weights = R2Plus1D_18_Weights.DEFAULT
        self.model = r2plus1d_18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # Input: (Batch, Seq, C, H, W) -> Needed: (Batch, C, Seq, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)


# ==========================================
# 3. Video Transformer (VideoMAE)
# ==========================================
class VideoTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoTransformer, self).__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Please run 'pip install transformers' to use this model.")

        # Using a lightweight VideoMAE model
        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        # Input: (Batch, Seq, C, H, W)
        # HuggingFace expects 'pixel_values' argument
        outputs = self.model(pixel_values=x)
        return outputs.logits


# ==========================================
# Factory Helper
# ==========================================
def get_model(model_name, num_classes=2):
    if model_name == "CNN-LSTM":
        return CNNLSTM(num_classes=num_classes)
    elif model_name == "3D CNN":
        return ResNet3D(num_classes=num_classes)
    elif model_name == "Video Transformer":
        return VideoTransformer(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")