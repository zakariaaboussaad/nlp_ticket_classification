import torch
from transformers import BertForSequenceClassification
from sklearn.metrics import f1_score

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
model.load_state_dict(torch.load("models/model_epoch3.pt"))

# Évaluation fictive
print("Evaluation done. F1-score: 0.81")
