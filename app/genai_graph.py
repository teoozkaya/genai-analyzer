from langgraph.graph import StateGraph
from app.prompts import summarization_prompt, cause_analysis_prompt, resolution_prompt
from typing import TypedDict
from langchain_core.runnables import RunnableLambda
import os
from langchain_openai import ChatOpenAI

def get_llm():
    model_name = os.getenv("LLM_MODEL", "phi-4")
    print(f"Connecting to LLM with model: {model_name}")
    return ChatOpenAI(
        model=model_name,
        base_url="http://host.docker.internal:1234/v1",
        api_key="lm-studio"
    )

class LogState(TypedDict):
    logs: str
    summary: str
    causes: str
    fixes: str

def summarize(state):
    if "logs" not in state:
        print("❌ 'logs' key not found in state:", state)
        raise ValueError("Missing 'logs' in state")
    prompt = summarization_prompt(state["logs"])
    return {"summary": get_llm().invoke(prompt).content}

def analyze(state):
    prompt = cause_analysis_prompt(state["summary"])
    return {"causes": get_llm().invoke(prompt).content}

def suggest(state):
    prompt = resolution_prompt(state["summary"])
    return {"fixes": get_llm().invoke(prompt).content}

def build_log_analysis_graph():
    builder = StateGraph(LogState)

    # Add nodes for each processing stage
    builder.add_node("summarize", RunnableLambda(summarize))
    builder.add_node("analyze", RunnableLambda(analyze))
    builder.add_node("suggest", RunnableLambda(suggest))

    # Define flow structure
    builder.set_entry_point("summarize")
    builder.add_edge("summarize", "analyze")
    builder.add_edge("analyze", "suggest")
    builder.set_finish_point("suggest")

    return builder.compile()
