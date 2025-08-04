def summarization_prompt(logs: str) -> str:
    return f"""You are a log summarizer.
Here are system logs:
{logs}

Summarize the most important issues and group them by severity.
"""

def cause_analysis_prompt(summary: str) -> str:
    return f"""Given this summary of system issues:

{summary}

Suggest likely root causes based on typical software failure scenarios."""

def resolution_prompt(summary: str) -> str:
    return f"""Given this summary of system issues:

{summary}

Suggest concrete resolution or mitigation steps that a developer can take."""
