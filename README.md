# 🔍 GenAI Log Analyzer

A modern log analysis tool that combines classic machine learning with GenAI capabilities to summarize, explain, and suggest fixes for system logs.

Built as a proof-of-concept aligned with the automotive domain — easily adaptable for diagnostics, DevOps, or internal IT tools.

---

## ⚙️ Tech Stack

- **FastAPI** – RESTful API for file upload and log processing  
- **PyTorch** – Lightweight classifier to predict severity (INFO, WARNING, ERROR)  
- **TfidfVectorizer** – Converts raw logs into numerical features  
- **LangGraph + LangChain** – Multi-step GenAI pipeline: summary → root cause → fixes  
- **LLM Backend** – Compatible with OpenAI or LM Studio (via `ChatOpenAI`)  

---

## 🧠 What It Does

| Step                  | Function                                               |
|-----------------------|--------------------------------------------------------|
| ✅ Upload log file     | Plain text file with system logs                      |
| ✅ ML Classification   | Assigns severity using trained PyTorch model          |
| ✅ Optional GenAI      | Toggle `use_genai=true` to enable GenAI pipeline      |
| ✅ Summarization       | LLM summarizes logs by severity                       |
| ✅ Root Cause & Fixes  | LLM suggests possible causes and resolution steps     |
| ✅ Stats               | Returns severity counts, percentages, and dominant type |

---

## 🚀 How to Run (Local or Docker)

### 1. Build the Docker image
```bash
docker build -t genai-log-analyzer .

### 2. Start the App with LM Studio
- First, make sure LM Studio is running with a compatible model:
- Launch LM Studio
- Load a model (e.g., phi-2, mistral, or deepseek-r1-distill-qwen-7b)
- Ensure the OpenAI-compatible API server is running in LM Studio (default port: 1234)
Either:
#### Option A: Set model in the Dockerfile
Edit this line in your Dockerfile:
ENV LLM_MODEL="deepseek-r1-distill-qwen-7b"
#### Option B: Set model dynamically at runtime
Use the -e flag:
docker run -p 8000:8000 -e LLM_MODEL="mistral" genai-log-analyzer

### 3. Test It
####In a second terminal window:
curl -X POST "http://localhost:8000/analyze?use_genai=true" \
     -F "file=@app/sample_logs/sample1-log.txt"

Depending on the model you are using this process can take between 1-10 minutes
####You can also try:
open this in your browser to use the Swagger UI:
http://localhost:8000/docs
to view the Swagger UI.

## 🧾 Sample Response (GenAI enabled)
{
  "status": "success",
  "result": {
    "statistics": {
      "INFO": 2,
      "WARNING": 1,
      "ERROR": 1,
      "dominant": "INFO"
    },
    "genai_analysis": {
      "summary": "Connected to DB. Job completed. Retry attempt failed. Timeout while calling API.",
      "causes": "API Timeout likely due to network or config error.",
      "fixes": "Check API key, increase timeout limit, retry with backoff."
    }
  }
}

## ✍️ Author Notes
This project was built as a GenAI job application proof-of-concept.
Strong focus on explainability and control
Easily extensible for alerts, metrics, or a frontend dashboard
Offline LLM support via LM Studio, or OpenAI API-ready

## 📦 API Summary
Endpoint	Method	Description
/analyze	POST	Upload a log file for analysis
/docs	GET	View interactive API docs (Swagger UI)
