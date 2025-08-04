FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-multipart

COPY app/ ./app/
COPY app/vectorizer.pkl app/log_model.pt app/label_encoder.pkl /app/app/

EXPOSE 8000

# Edit this line accordingly with the model you are running
ENV LLM_MODEL="phi-4"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

