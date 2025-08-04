from fastapi import FastAPI, File, UploadFile, Query
from app.analyzer import process_log_file
import logging

app = FastAPI()


@app.post("/analyze")
async def analyze_file(
        file: UploadFile = File(...),
        use_genai: bool = Query(default=False, description="Whether to include GenAI insights")
    ):
    """
    Endpoint to upload a log file and analyze it.
    If `use_genai=true`, invokes the GenAI pipeline for enhanced output.
    """

    try:
        log_text = (await file.read()).decode("utf-8")
        result = process_log_file(log_text, use_genai=use_genai)
        return {"status": "success", "result": result}
    except Exception as e:
        logging.exception("🔥 Error during log analysis")
        return {"status": "error", "message": str(e)}
