import torch
import torch.nn as nn
import joblib
import os

BASE_DIR = os.path.dirname(__file__)  # resolves to /app/app

vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

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

model = LogClassifier(input_dim=len(vectorizer.get_feature_names_out()), num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "log_model.pt")))
model.eval()

def predict_severity(log_line: str, return_confidence=False):
    """
    Predicts the severity level of a given log line using the trained model.
    Returns one of: 'INFO', 'WARNING', 'ERROR'.
    """
    vec = vectorizer.transform([log_line]).toarray()
    tensor = torch.tensor(vec, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()

    label = label_encoder.inverse_transform([predicted_idx])[0]
    if return_confidence:
        confidence = probs[0][predicted_idx].item()
        return label, confidence
    return label
