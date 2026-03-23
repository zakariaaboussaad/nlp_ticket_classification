import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def save_model(model, path):
    """Save model state dict"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, **kwargs):
    """Load model state dict"""
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path))
    return model

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def get_classification_report(y_true, y_pred, target_names=None):
    """Get detailed classification report"""
    return classification_report(y_true, y_pred, target_names=target_names)
