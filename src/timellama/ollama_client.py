"""Ollama integration for LLM-powered formatting and suggestions."""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import ollama
from ollama import ResponseError


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


# ---------------------------------------------------------------------------
# New formatting functions for Phase 1: Raw JSON -> Ollama -> Display
# ---------------------------------------------------------------------------


def format_for_display(
    raw_data: dict | list | str, format_type: str, context: str = ""
) -> dict[str, Any]:
    """Pass raw MCP data to Ollama for intelligent formatting.

    Args:
        raw_data: Raw JSON data from MCP response
        format_type: Type of formatting ("status", "hours", "events", "entries")
        context: Additional context for the LLM

    Returns:
        {"display_text": str, "extracted": dict, "success": bool, "error": str | None}
    """
    available, error = check_ollama_available()
    if not available:
        return _format_fallback(raw_data, format_type)

    config = get_config()

    # Convert to JSON string if needed
    if isinstance(raw_data, (dict, list)):
        data_str = json.dumps(raw_data, indent=2, default=str)
    else:
        data_str = str(raw_data)

    prompts = {
        "status": f"""Format this time tracking status data for display.
Extract key info: date, logged hours, entry ID, event count.
Return a concise summary suitable for terminal display.

Data:
{data_str}

{f"Context: {context}" if context else ""}

Return JSON with format:
{{"display_text": "formatted summary for terminal", "extracted": {{"date": "...", "hours": ..., "entry_id": "...", "event_count": ...}}}}""",
        "hours": f"""Format this hours data for display as a summary table.
Handle different field names: 'hours' or 'time' (minutes), 'total_hours', etc.

Data:
{data_str}

{f"Context: {context}" if context else ""}

Return JSON with format:
{{"display_text": "formatted hours summary", "extracted": {{"total_hours": ..., "client_hours": ..., "internal_hours": ..., "entries": [...]}}}}""",
        "events": f"""Format these calendar events for display.
Create a clean list with event names and times.

Data:
{data_str}

{f"Context: {context}" if context else ""}

Return JSON with format:
{{"display_text": "formatted event list", "extracted": {{"events": [{{"summary": "...", "start": "...", "end": "..."}}], "count": ...}}}}""",
        "entries": f"""Format these time entries for display.
Handle different field names: 'time' (minutes) or 'hours', 'date' might be None.

Data:
{data_str}

{f"Context: {context}" if context else ""}

Return JSON with format:
{{"display_text": "formatted entries", "extracted": {{"entries": [...], "total_minutes": ...}}}}""",
    }

    prompt = prompts.get(format_type, prompts["status"])

    try:
        client = ollama.Client(host=config.host)
        response = client.generate(model=config.model, prompt=prompt, format="json")
        result = json.loads(response.response.strip())

        if isinstance(result, dict) and "display_text" in result:
            result["success"] = True
            result["error"] = None
            return result
        else:
            return _format_fallback(raw_data, format_type)

    except (ResponseError, json.JSONDecodeError, Exception) as e:
        fallback = _format_fallback(raw_data, format_type)
        fallback["error"] = str(e)
        return fallback


def extract_action_data(
    raw_data: dict | list | str, action: str, context: str = ""
) -> dict[str, Any]:
    """Extract specific fields needed for CRUD operations.

    Args:
        raw_data: Raw JSON data from MCP response
        action: Action type ("get_entry_id", "get_note", "get_hours", "filter_events")
        context: Additional context (e.g., filter criteria)

    Returns:
        {"data": extracted_value, "success": bool, "error": str | None}
    """
    available, error = check_ollama_available()
    if not available:
        return _extract_fallback(raw_data, action, context)

    config = get_config()

    if isinstance(raw_data, (dict, list)):
        data_str = json.dumps(raw_data, indent=2, default=str)
    else:
        data_str = str(raw_data)

    prompts = {
        "get_entry_id": f"""Extract the time entry ID from this data.
Look for fields like 'id', 'entry_id', 'time_entry_id'.

Data:
{data_str}

Return JSON: {{"data": "the_id_value", "found": true}} or {{"data": null, "found": false}}""",
        "get_note": f"""Extract the note/description from this time entry data.
Look for 'note', 'description', 'notes' fields.

Data:
{data_str}

Return JSON: {{"data": "the note content", "found": true}} or {{"data": null, "found": false}}""",
        "get_hours": f"""Extract hours/time from this data.
Handle 'hours' (float) or 'time' (minutes). Convert minutes to hours if needed.

Data:
{data_str}

Return JSON: {{"data": hours_as_float, "found": true, "was_minutes": true/false}}""",
        "filter_events": f"""Filter these events based on criteria.
{f"Filter: {context}" if context else "Remove busy/private events."}

Data:
{data_str}

Return JSON: {{"data": [filtered_events], "removed_count": N}}""",
    }

    prompt = prompts.get(action, prompts["get_entry_id"])

    try:
        client = ollama.Client(host=config.host)
        response = client.generate(model=config.model, prompt=prompt, format="json")
        result = json.loads(response.response.strip())

        if isinstance(result, dict):
            result["success"] = True
            result["error"] = None
            return result
        else:
            return _extract_fallback(raw_data, action, context)

    except (ResponseError, json.JSONDecodeError, Exception) as e:
        fallback = _extract_fallback(raw_data, action, context)
        fallback["error"] = str(e)
        return fallback


def _format_fallback(raw_data: dict | list | str, format_type: str) -> dict[str, Any]:
    """Fallback formatting when Ollama is unavailable."""
    extracted = {}
    display_text = ""

    try:
        if isinstance(raw_data, str):
            try:
                raw_data = json.loads(raw_data)
            except json.JSONDecodeError:
                return {
                    "display_text": str(raw_data)[:500],
                    "extracted": {},
                    "success": True,
                    "error": None,
                }

        if format_type == "status":
            # Extract from various possible structures
            if isinstance(raw_data, dict):
                extracted["date"] = raw_data.get("date")
                extracted["hours"] = raw_data.get("hours") or raw_data.get("time", 0) / 60
                extracted["entry_id"] = raw_data.get("id") or raw_data.get("entry_id")
                events = raw_data.get("events", [])
                extracted["event_count"] = len(events) if isinstance(events, list) else 0

            hours = extracted.get("hours", 0)
            display_text = f"Date: {extracted.get('date', 'today')} | Hours: {hours:.1f}h | Events: {extracted.get('event_count', 0)}"

        elif format_type == "hours":
            if isinstance(raw_data, dict):
                # Handle both 'hours' and 'time' fields
                total = raw_data.get("total_hours") or raw_data.get("hours", 0)
                if raw_data.get("time"):
                    total = raw_data["time"] / 60
                extracted["total_hours"] = total
                extracted["client_hours"] = raw_data.get("client_hours", 0)
                extracted["internal_hours"] = raw_data.get("internal_hours", 0)
                extracted["entries"] = raw_data.get("entries", [])

            display_text = f"Total: {extracted.get('total_hours', 0):.1f}h"

        elif format_type == "events":
            events = []
            if isinstance(raw_data, dict):
                events = raw_data.get("events", [])
            elif isinstance(raw_data, list):
                events = raw_data

            extracted["events"] = events
            extracted["count"] = len(events)

            if events:
                lines = [f"• {e.get('summary', 'Event')}" for e in events[:5]]
                display_text = "\n".join(lines)
                if len(events) > 5:
                    display_text += f"\n... and {len(events) - 5} more"
            else:
                display_text = "No events"

        elif format_type == "entries":
            entries = []
            if isinstance(raw_data, dict):
                entries = raw_data.get("entries", [])
            elif isinstance(raw_data, list):
                entries = raw_data

            total_minutes = sum(e.get("time", 0) for e in entries if isinstance(e, dict))
            extracted["entries"] = entries
            extracted["total_minutes"] = total_minutes

            display_text = f"{len(entries)} entries, {total_minutes / 60:.1f}h total"

        else:
            display_text = json.dumps(raw_data, indent=2, default=str)[:500]

    except Exception:
        display_text = str(raw_data)[:500]

    return {
        "display_text": display_text,
        "extracted": extracted,
        "success": True,
        "error": None,
    }


def _extract_fallback(
    raw_data: dict | list | str, action: str, context: str = ""
) -> dict[str, Any]:
    """Fallback extraction when Ollama is unavailable."""
    try:
        if isinstance(raw_data, str):
            try:
                raw_data = json.loads(raw_data)
            except json.JSONDecodeError:
                return {"data": None, "success": False, "error": "Invalid JSON"}

        if action == "get_entry_id":
            if isinstance(raw_data, dict):
                entry_id = raw_data.get("id") or raw_data.get("entry_id") or raw_data.get("time_entry_id")
                return {"data": entry_id, "found": entry_id is not None, "success": True, "error": None}
            elif isinstance(raw_data, list) and raw_data:
                entry_id = raw_data[0].get("id") if isinstance(raw_data[0], dict) else None
                return {"data": entry_id, "found": entry_id is not None, "success": True, "error": None}

        elif action == "get_note":
            if isinstance(raw_data, dict):
                note = raw_data.get("note") or raw_data.get("description") or raw_data.get("notes")
                return {"data": note, "found": note is not None, "success": True, "error": None}

        elif action == "get_hours":
            if isinstance(raw_data, dict):
                hours = raw_data.get("hours")
                time_minutes = raw_data.get("time")
                if hours is not None:
                    return {"data": float(hours), "found": True, "was_minutes": False, "success": True, "error": None}
                elif time_minutes is not None:
                    return {"data": float(time_minutes) / 60, "found": True, "was_minutes": True, "success": True, "error": None}

        elif action == "filter_events":
            events = []
            if isinstance(raw_data, dict):
                events = raw_data.get("events", [])
            elif isinstance(raw_data, list):
                events = raw_data

            skip_keywords = {"busy", "private", "blocked", "focus time", "lunch"}
            filtered = [
                e for e in events
                if isinstance(e, dict) and e.get("summary", "").lower() not in skip_keywords
            ]
            return {"data": filtered, "removed_count": len(events) - len(filtered), "success": True, "error": None}

        return {"data": None, "success": False, "error": f"Unknown action: {action}"}

    except Exception as e:
        return {"data": None, "success": False, "error": str(e)}


def format_events_to_html_from_raw(raw_events: list[dict] | dict | str) -> str:
    """Format raw event data to HTML list using Ollama.

    Falls back to simple formatting if Ollama is unavailable.
    This is the new version that works with raw JSON data.
    """
    # Parse if string
    if isinstance(raw_events, str):
        try:
            raw_events = json.loads(raw_events)
        except json.JSONDecodeError:
            return "<ul><li>No meetings today</li></ul>"

    # Extract events list
    if isinstance(raw_events, dict):
        events = raw_events.get("events", [])
    else:
        events = raw_events

    if not events:
        return "<ul><li>No meetings today</li></ul>"

    # Check Ollama availability
    available, error = check_ollama_available()
    if not available:
        return _format_events_simple_raw(events)

    config = get_config()

    # Prepare events for LLM
    events_text = "\n".join(
        [f"- {e.get('summary', 'Event')} ({e.get('start', '')} - {e.get('end', '')})" for e in events]
    )

    prompt = f"""Convert the following calendar events into an HTML unordered list.
Use ONLY the actual event names provided below. Do NOT use example names.
Output ONLY the HTML, no explanation.

ACTUAL EVENTS TO FORMAT:
{events_text}

OUTPUT FORMAT:
<ul>
<li>[first event name from above]</li>
<li>[second event name from above]</li>
</ul>"""

    try:
        client = ollama.Client(host=config.host)
        response = client.generate(model=config.model, prompt=prompt)
        html = response.response.strip()

        if "<ul>" in html and "</ul>" in html:
            return html
        else:
            return _format_events_simple_raw(events)
    except (ResponseError, Exception):
        return _format_events_simple_raw(events)


def _format_events_simple_raw(events: list[dict]) -> str:
    """Simple HTML formatting from raw event data without LLM."""
    if not events:
        return "<ul><li>No meetings today</li></ul>"

    # Filter out "busy" events
    skip_keywords = {"busy", "private", "blocked"}
    filtered = [
        e for e in events
        if isinstance(e, dict) and e.get("summary", "").lower() not in skip_keywords
        and not e.get("summary", "").lower().startswith("busy")
    ]

    if not filtered:
        return "<ul><li>No meetings today</li></ul>"

    items = "\n".join([f"<li>{e.get('summary', 'Event')}</li>" for e in filtered])
    return f"<ul>\n{items}\n</ul>"


# Keep the old function for backward compatibility
def format_events_to_html(events: list) -> str:
    """Format calendar events to HTML list using Ollama.

    Falls back to simple formatting if Ollama is unavailable.
    Accepts both CalendarEvent objects and raw dicts.
    """
    if not events:
        return "<ul><li>No meetings today</li></ul>"

    # Convert to raw dict format if needed (for CalendarEvent objects)
    raw_events = []
    for e in events:
        if hasattr(e, "summary"):
            # CalendarEvent object
            raw_events.append({
                "summary": e.summary,
                "start": e.start.strftime("%H:%M") if hasattr(e.start, "strftime") else str(e.start),
                "end": e.end.strftime("%H:%M") if hasattr(e.end, "strftime") else str(e.end),
            })
        elif isinstance(e, dict):
            raw_events.append(e)

    return format_events_to_html_from_raw(raw_events)


def _format_events_simple(events: list) -> str:
    """Simple HTML formatting without LLM.

    Accepts both CalendarEvent objects and raw dicts.
    """
    if not events:
        return "<ul><li>No meetings today</li></ul>"

    # Handle both CalendarEvent objects and dicts
    filtered = []
    for e in events:
        if hasattr(e, "summary"):
            summary = e.summary
        elif isinstance(e, dict):
            summary = e.get("summary", "")
        else:
            continue

        if summary.lower() not in ("busy", "private", "blocked"):
            filtered.append(summary)

    if not filtered:
        return "<ul><li>No meetings today</li></ul>"

    items = "\n".join([f"<li>{s}</li>" for s in filtered])
    return f"<ul>\n{items}\n</ul>"


def generate_suggestions(
    events: list, existing_note: str | None = None
) -> list[str]:
    """Generate suggestions for additional time entry items.

    Accepts both CalendarEvent objects and raw dicts.

    Returns a list of suggested items to add.
    """
    available, _ = check_ollama_available()
    if not available:
        return []

    config = get_config()

    # Handle both CalendarEvent objects and dicts
    event_summaries = []
    for e in events:
        if hasattr(e, "summary"):
            event_summaries.append(e.summary)
        elif isinstance(e, dict):
            event_summaries.append(e.get("summary", "Event"))

    events_text = (
        "\n".join([f"- {s}" for s in event_summaries]) if event_summaries else "No events"
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
