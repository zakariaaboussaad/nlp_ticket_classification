import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AdamW
from src.data_preprocessing import load_and_split, tokenize_texts

# Charger dataset fictif
train, val, test = load_and_split("data/tickets.csv")
inputs = tokenize_texts(train['text'])
labels = torch.tensor(train['label'].values)

dataset = TensorDataset(inputs['input_ids'], labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    for batch in loader:
        input_ids, labels = batch
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} done")

torch.save(model.state_dict(), "models/model_epoch3.pt")
