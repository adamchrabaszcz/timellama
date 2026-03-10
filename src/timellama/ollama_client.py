"""Ollama integration for LLM-powered formatting and suggestions."""

import os
from dataclasses import dataclass
from datetime import datetime

import ollama
from ollama import ResponseError

from timellama.mcp_client import CalendarEvent


@dataclass
class OllamaConfig:
    """Ollama configuration."""

    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"


def get_config() -> OllamaConfig:
    """Get Ollama configuration from environment."""
    return OllamaConfig(
        model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )


def check_ollama_available() -> tuple[bool, str]:
    """Check if Ollama is available and running.

    Returns:
        Tuple of (is_available, error_message)
    """
    config = get_config()
    try:
        client = ollama.Client(host=config.host)
        client.list()
        return True, ""
    except Exception as e:
        return False, str(e)


def check_model_available(model: str | None = None) -> tuple[bool, str]:
    """Check if the specified model is available.

    Returns:
        Tuple of (is_available, error_message)
    """
    config = get_config()
    model_name = model or config.model

    try:
        client = ollama.Client(host=config.host)
        models = client.list()
        available_models = [m.model for m in models.models]

        # Check for exact match or partial match (model:tag)
        for available in available_models:
            if model_name in available or available.startswith(model_name.split(":")[0]):
                return True, ""

        return False, f"Model '{model_name}' not found. Available: {available_models}"
    except Exception as e:
        return False, str(e)


def format_events_to_html(events: list[CalendarEvent]) -> str:
    """Format calendar events to HTML list using Ollama.

    Falls back to simple formatting if Ollama is unavailable.
    """
    if not events:
        return "<ul><li>No meetings today</li></ul>"

    # Check Ollama availability
    available, error = check_ollama_available()
    if not available:
        return _format_events_simple(events)

    config = get_config()

    # Prepare events for LLM
    events_text = "\n".join(
        [
            f"- {e.summary} ({e.start.strftime('%H:%M')}-{e.end.strftime('%H:%M')})"
            for e in events
        ]
    )

    prompt = f"""Convert these calendar events into a clean HTML unordered list for a time tracking note.
Keep it concise - just the event name, no times needed.
Remove any "Busy" placeholders or private events.
Group similar meetings if appropriate.

Events:
{events_text}

Return ONLY the HTML <ul> list, nothing else. Example format:
<ul>
<li>Team standup</li>
<li>Code review session</li>
</ul>"""

    try:
        client = ollama.Client(host=config.host)
        response = client.generate(model=config.model, prompt=prompt)
        html = response.response.strip()

        # Validate it's HTML
        if "<ul>" in html and "</ul>" in html:
            return html
        else:
            return _format_events_simple(events)
    except (ResponseError, Exception):
        return _format_events_simple(events)


def _format_events_simple(events: list[CalendarEvent]) -> str:
    """Simple HTML formatting without LLM."""
    if not events:
        return "<ul><li>No meetings today</li></ul>"

    # Filter out "busy" events
    filtered = [
        e for e in events if e.summary.lower() not in ("busy", "private", "blocked")
    ]

    if not filtered:
        return "<ul><li>No meetings today</li></ul>"

    items = "\n".join([f"<li>{e.summary}</li>" for e in filtered])
    return f"<ul>\n{items}\n</ul>"


def generate_suggestions(
    events: list[CalendarEvent], existing_note: str | None = None
) -> list[str]:
    """Generate suggestions for additional time entry items.

    Returns a list of suggested items to add.
    """
    available, _ = check_ollama_available()
    if not available:
        return []

    config = get_config()

    events_text = (
        "\n".join([f"- {e.summary}" for e in events]) if events else "No events"
    )
    existing_text = existing_note or "None"

    prompt = f"""Based on these calendar events and existing time log, suggest 2-3 additional work items
that a developer might have done but didn't have calendar events for.

Calendar events:
{events_text}

Current time log:
{existing_text}

Common developer tasks: code review, debugging, documentation, PR reviews, slack discussions,
email, planning, research, deployment, testing.

Return ONLY a JSON array of strings with short task descriptions. Example:
["Code review", "Slack discussions", "Documentation updates"]"""

    try:
        client = ollama.Client(host=config.host)
        response = client.generate(model=config.model, prompt=prompt)
        text = response.response.strip()

        # Parse JSON array
        import json

        suggestions = json.loads(text)
        if isinstance(suggestions, list):
            return [str(s) for s in suggestions[:5]]
        return []
    except Exception:
        return []


def summarize_work(entries_text: str, hours_data: dict | None = None) -> str:
    """Generate a summary of work entries for invoice approval.

    Args:
        entries_text: Text description of work entries
        hours_data: Optional hours breakdown data

    Returns:
        Human-readable summary of work
    """
    available, _ = check_ollama_available()
    if not available:
        return entries_text

    config = get_config()

    hours_context = ""
    if hours_data:
        hours_context = f"""
Hours breakdown:
- Total: {hours_data.get('total_hours', 'N/A')}h
- Client work: {hours_data.get('client_hours', 'N/A')}h
- Internal: {hours_data.get('internal_hours', 'N/A')}h
"""

    prompt = f"""Summarize this employee's work for invoice review. Be concise but comprehensive.
Group similar activities together. Highlight key deliverables.

{hours_context}

Work entries:
{entries_text}

Return a brief professional summary (2-4 sentences) suitable for invoice approval."""

    try:
        client = ollama.Client(host=config.host)
        response = client.generate(model=config.model, prompt=prompt)
        return response.response.strip()
    except Exception:
        return entries_text


def parse_chat_input(user_input: str, context: str | None = None) -> dict:
    """Parse user chat input to determine intent and extract data.

    Args:
        user_input: Raw user input
        context: Optional context about current state

    Returns:
        Dict with 'intent' and 'data' keys
    """
    # Simple keyword matching for common commands
    input_lower = user_input.lower().strip()

    if input_lower in ("quit", "exit", "q"):
        return {"intent": "quit", "data": None}

    if input_lower in ("sync", "update", "refresh"):
        return {"intent": "sync", "data": None}

    if input_lower in ("show", "status", "current"):
        return {"intent": "show", "data": None}

    if input_lower.startswith("add "):
        item = user_input[4:].strip()
        return {"intent": "add", "data": item}

    if input_lower.startswith("remove ") or input_lower.startswith("delete "):
        item = user_input.split(" ", 1)[1].strip()
        return {"intent": "remove", "data": item}

    # For complex inputs, try Ollama
    available, _ = check_ollama_available()
    if not available:
        return {"intent": "unknown", "data": user_input}

    config = get_config()

    prompt = f"""Classify this user input for a time tracking assistant.

Input: "{user_input}"

Return ONLY a JSON object with:
- intent: one of "sync", "show", "add", "remove", "help", "quit", "unknown"
- data: extracted data if relevant (e.g., item to add/remove)

Example: {{"intent": "add", "data": "code review"}}"""

    try:
        client = ollama.Client(host=config.host)
        response = client.generate(model=config.model, prompt=prompt)
        text = response.response.strip()

        import json

        result = json.loads(text)
        if isinstance(result, dict) and "intent" in result:
            return result
        return {"intent": "unknown", "data": user_input}
    except Exception:
        return {"intent": "unknown", "data": user_input}
