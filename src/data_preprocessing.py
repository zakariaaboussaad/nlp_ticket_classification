import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch

def load_and_split(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    return train, val, test

def tokenize_texts(texts, tokenizer_name='bert-base-uncased', max_len=128):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
