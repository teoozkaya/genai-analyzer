from app.model import predict_severity 
from app.genai_graph import build_log_analysis_graph

from collections import defaultdict

graph = build_log_analysis_graph()

def process_log_file(log_text: str, use_genai: bool = False) -> dict:
    """
    Analyze a raw log file and return structured insights.
    - Predicts severity using ML
    - Computes statistics
    - Optionally adds GenAI summary, causes, and fixes
    """
    lines = [line.strip() for line in log_text.strip().split("\n") if line.strip()]
    results = []

    for line in lines:
        severity, confidence = predict_severity(line, return_confidence=True)
        results.append({
            "line": line,
            "predicted_severity": severity,
            "confidence": round(confidence, 4),
        })

    grouped_logs = group_by_severity(results)
    stats = compute_log_statistics(results)

    genai_results = None
    if use_genai:
        genai_results = graph.invoke({"logs": log_text})

    return {
        "grouped_logs": grouped_logs,
        "statistics": stats,
        "genai_analysis": genai_results if use_genai else None,
    }


def group_similar_logs(log_lines: list[dict], num_buckets=3):
    groups = defaultdict(list)
    for line in log_lines:
        label = line["predicted_severity"]["label"]
        groups[label].append(line)
    return groups



def compute_log_statistics(results: list[dict]):
    severity_counts = defaultdict(int)
    for item in results:
        severity_counts[item["predicted_severity"]] += 1

    total = len(results)
    stats = {
        "total_logs": total,
        "severity_counts": dict(severity_counts),
        "severity_percentages": {
            k: round((v / total) * 100, 2) for k, v in severity_counts.items()
        },
        "dominant_severity": max(severity_counts, key=severity_counts.get)
    }
    return stats

def group_by_severity(results: list[dict]):
    grouped = defaultdict(list)
    for entry in results:
        grouped[entry["predicted_severity"]].append(entry)
    return dict(grouped)
