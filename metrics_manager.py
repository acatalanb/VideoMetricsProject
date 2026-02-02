import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score


class MetricsManager:
    def __init__(self, model_name):
        # Create a safe filename safe string (e.g. "3D CNN" -> "3D_CNN")
        self.safe_name = model_name.replace(" ", "_")
        self.metrics_file = f"metrics_{self.safe_name}.json"
        self.cm_file = f"cm_{self.safe_name}.png"
        self.roc_file = f"roc_{self.safe_name}.png"
        self.start_time = None
        self.end_time = None

    def start_training_timer(self):
        self.start_time = time.time()

    def stop_training_timer(self):
        self.end_time = time.time()
        return self.end_time - self.start_time

    def compute_metrics(self, y_true, y_pred, y_probs):
        """Calculates a comprehensive dictionary of metrics."""

        # 1. Confusion Matrix elements
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 2. Derived Metrics
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)  # Same as Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 3. AUC calculation
        try:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.5
            fpr, tpr = [0, 1], [0, 1]

        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1,
            "auc": roc_auc,
            "confusion_matrix": cm.tolist(),  # Convert to list for JSON
            "roc_data": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        }

    def save_metrics(self, metrics_dict, training_time):
        """Saves metrics and training time to JSON."""
        metrics_dict["training_time_seconds"] = training_time
        metrics_dict["training_timestamp"] = time.ctime()

        with open(self.metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics saved to {self.metrics_file}")

    def load_metrics(self):
        """Loads the JSON metrics from disk."""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, "r") as f:
                return json.load(f)
        return None

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Abnormal'],
                    yticklabels=['Normal', 'Abnormal'])
        plt.title(f'Confusion Matrix: {self.safe_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.cm_file)
        plt.close()

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {self.safe_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.roc_file)
        plt.close()