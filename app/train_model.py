import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import joblib
import numpy as np
from log_dataset import log_data

data = log_data

texts, labels = zip(*data)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

joblib.dump(label_encoder, "label_encoder.pkl")

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()

joblib.dump(vectorizer, "vectorizer.pkl")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size = 0.2, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor), batch_size=4, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_tensor, y_val_tensor), batch_size=4, shuffle = False
)


class LogClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        return self.model(x)

model = LogClassifier(input_dim=X.shape[1], num_classes=len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training\n")
for epoch in range(30):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

model.eval()
with torch.no_grad():
    val_preds = model(X_val_tensor).argmax(dim=1)
    print("\nValidation Results:")
    print(classification_report(y_val, val_preds, target_names=label_encoder.classes_))

torch.save(model.state_dict(), "app/log_model.pt")
print("\nModel, vectorizer, and label encoder saved.")
