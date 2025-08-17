# 🔍 GenAI Log Analyzer

A modern log analysis tool that combines classic machine learning with GenAI capabilities to summarize, explain, and suggest fixes for system logs.

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
| Upload log file     | Plain text file with system logs                      |
| ML Classification   | Assigns severity using trained PyTorch model          |
| Optional GenAI      | Toggle `use_genai=true` to enable GenAI pipeline      |
| Summarization       | LLM summarizes logs by severity                       |
| Root Cause & Fixes  | LLM suggests possible causes and resolution steps     |
| Stats               | Returns severity counts, percentages, and dominant type |

---

## 🚀 How to Run (Local or Docker)

### 1. Clone the repository: 
Copy this into the terminal
```bash
git clone https://github.com/teoozkaya/genai-analyzer.git
```
### 2. Change folder: 
Go to the directory: 
```bash
cd genai-anaylzer
```
### 3. Build the Docker image
Make sure you have Docker installed and running on your system. Then run:
```bash
docker build -t genai-log-analyzer .
```
### 4. Start the App with LM Studio
- First, make sure LM Studio is running with a compatible model:  
- Launch LM Studio  
- Load a model (e.g., phi-2, mistral, or deepseek-r1-distill-qwen-7b)  
- Ensure the OpenAI-compatible API server is running in LM Studio (default port: 1234)  
Either:
#### Option A: Set model in the Dockerfile
Edit this line in your Dockerfile:
```bash
ENV LLM_MODEL="phi-4"
```
#### Option B: Set model dynamically at runtime
Use the -e flag:
```bash
docker run -p 8000:8000 -e LLM_MODEL="YOUR-MODEL" genai-log-analyzer
```
### 5. Test It
There is a ready sample log file to analyze
#### In a second terminal window:
```bash
curl -X POST "http://localhost:8000/analyze?use_genai=true" \
     -F "file=@genai-analyzer/app/sample_logs/sample1-log.txt"
```

Depending on the model you are using this process can take between 1-10 minutes
#### You can also try:
open this in your browser to use the Swagger UI:  
http://localhost:8000/docs  
to view the Swagger UI.  

## 🧾 Sample Response (GenAI enabled)
```bash
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
```
## ✍️ Author Notes
This project was built as a GenAI job application proof-of-concept.  
Strong focus on explainability and control.  
Easily extensible for alerts, metrics, or a frontend dashboard.  
Offline LLM support via LM Studio, or OpenAI API-ready.  

## 📦 API Summary
Endpoint	Method	Description
/analyze	POST	Upload a log file for analysis  
/docs	GET	View interactive API docs (Swagger UI)  
