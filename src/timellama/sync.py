"""Core sync logic for calendar to Productive time entries."""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from timellama.mcp_client import CalendarEvent, MCPClient, TimeEntry
from timellama.ollama_client import (
    extract_action_data,
    format_events_to_html,
    format_events_to_html_from_raw,
    format_for_display,
    generate_suggestions,
)


@dataclass
class SyncResult:
    """Result of a sync operation."""

    success: bool
    action: str  # "created", "updated", "skipped", "error"
    message: str
    time_entry_id: str | None = None
    events_count: int = 0
    note_preview: str | None = None


async def sync_today(
    client: MCPClient,
    dry_run: bool = False,
    include_suggestions: bool = False,
) -> SyncResult:
    """Sync today's calendar events to Productive time entry.

    Uses raw JSON methods and Ollama formatters for resilient parsing.

    Args:
        client: Connected MCP client
        dry_run: If True, don't actually create/update entries
        include_suggestions: If True, include LLM-generated suggestions

    Returns:
        SyncResult with operation details
    """
    today = date.today()
    today_str = today.isoformat()

    # Get today's calendar events - try raw method first
    raw_events = None
    events = []
    calendar_error = None
    try:
        raw_events = await client.get_events_today_raw()

        # Check for error response
        if isinstance(raw_events, dict) and "error" in raw_events:
            calendar_error = raw_events["error"]
        else:
            # Extract events list from various response formats
            if isinstance(raw_events, dict):
                events_list = raw_events.get("events", [])
            elif isinstance(raw_events, list):
                events_list = raw_events
            else:
                events_list = []

            # Convert to CalendarEvent-like dicts for filtering
            events = events_list

    except Exception as e:
        calendar_error = str(e)
        # Fallback to old method
        try:
            event_objects = await client.get_events_today()
            calendar_error = None  # Clear if fallback works
            events = [
                {
                    "summary": e.summary,
                    "start": e.start.isoformat() if hasattr(e.start, "isoformat") else str(e.start),
                    "end": e.end.isoformat() if hasattr(e.end, "isoformat") else str(e.end),
                }
                for e in event_objects
            ]
        except Exception as e2:
            return SyncResult(
                success=False,
                action="error",
                message=f"Failed to fetch calendar events: {e2}",
            )

    # If calendar had an error, report it
    if calendar_error:
        return SyncResult(
            success=False,
            action="error",
            message=f"Calendar error: {calendar_error}",
        )

    # Filter out "busy" events using deterministic filter
    # (Ollama filtering was inconsistent, so we use simple keyword matching)
    filtered_events = _filter_events_raw(events)

    # Format events to HTML using raw formatter
    html_note = format_events_to_html_from_raw(filtered_events)

    # Optionally add suggestions
    if include_suggestions and filtered_events:
        # Convert to CalendarEvent for suggestion generator (backward compat)
        try:
            from timellama.mcp_client import CalendarEvent
            from datetime import datetime

            event_objects = []
            for e in filtered_events:
                if isinstance(e, dict):
                    event_objects.append(CalendarEvent(
                        summary=e.get("summary", "Event"),
                        start=datetime.fromisoformat(e.get("start", datetime.now().isoformat())),
                        end=datetime.fromisoformat(e.get("end", datetime.now().isoformat())),
                    ))
            suggestions = generate_suggestions(event_objects, html_note)
            if suggestions:
                html_note = html_note.replace(
                    "</ul>", f"<li><em>Other: {', '.join(suggestions)}</em></li>\n</ul>"
                )
        except Exception:
            pass  # Skip suggestions on error

    # Check for existing time entry today - try raw method first
    today_entry = None
    entry_id = None
    try:
        raw_entries = await client.get_time_entries_raw(after=today, before=today)

        entries_list = []
        if isinstance(raw_entries, dict):
            entries_list = raw_entries.get("entries", raw_entries.get("data", []))
            if not isinstance(entries_list, list):
                entries_list = [raw_entries] if raw_entries.get("id") else []
        elif isinstance(raw_entries, list):
            entries_list = raw_entries

        # Find today's entry
        for entry in entries_list:
            if isinstance(entry, dict):
                entry_date = entry.get("date")
                if entry_date is None or entry_date == today_str or str(entry_date).startswith(today_str):
                    today_entry = entry
                    # Parse numeric ID from report format: "time-entry-report-...-137120906-hash"
                    full_id = entry.get("id", "")
                    entry_id = _extract_numeric_id(full_id)
                    break

    except Exception:
        # Fallback to old method
        try:
            entries = await client.get_time_entries(after=today, before=today)
            today_entries = [e for e in entries if e.date == today_str]
            if today_entries:
                today_entry = {"id": today_entries[0].id}
                entry_id = today_entries[0].id
        except Exception as e:
            return SyncResult(
                success=False,
                action="error",
                message=f"Failed to fetch existing entries: {e}",
            )

    if dry_run:
        action = "would update" if today_entry else "would create"
        return SyncResult(
            success=True,
            action="skipped",
            message=f"Dry run: {action} time entry with {len(filtered_events)} events",
            events_count=len(filtered_events),
            note_preview=html_note,
        )

    # Create or update entry
    if today_entry and entry_id:
        # Update existing entry
        try:
            result = await client.update_time_entry(entry_id, note=html_note)
            return SyncResult(
                success=True,
                action="updated",
                message=f"Updated time entry with {len(filtered_events)} events",
                time_entry_id=entry_id,
                events_count=len(filtered_events),
                note_preview=html_note,
            )
        except Exception as e:
            return SyncResult(
                success=False,
                action="error",
                message=f"Failed to update entry: {e}",
            )
    else:
        # Create new entry (default 8 hours = 480 minutes)
        try:
            result = await client.create_time_entry(
                date_str=today_str,
                time_minutes=480,
                note=html_note,
            )

            # Check for service selection needed
            if isinstance(result, dict) and result.get("needs_service_selection"):
                services = result.get("available_services", [])
                service_list = "\n".join(
                    f"  - {s.get('service_name')} (ID: {s.get('service_id')}, used {s.get('count')}x)"
                    for s in services[:5]
                )
                return SyncResult(
                    success=False,
                    action="error",
                    message=f"Multiple services found. Set PRODUCTIVE_SERVICE_ID to one of:\n{service_list}",
                )

            # Check for error in result
            if isinstance(result, dict) and (result.get("error") or ("raw" in result and "Error" in str(result.get("raw", "")))):
                return SyncResult(
                    success=False,
                    action="error",
                    message=str(result["raw"]),
                )

            # Extract entry ID from result using Ollama extraction
            new_entry_id = None
            if isinstance(result, dict):
                extracted = extract_action_data(result, "get_entry_id")
                if extracted.get("success") and extracted.get("data"):
                    new_entry_id = extracted["data"]
                else:
                    new_entry_id = result.get("id")

            return SyncResult(
                success=True,
                action="created",
                message=f"Created time entry with {len(filtered_events)} events",
                time_entry_id=new_entry_id,
                events_count=len(filtered_events),
                note_preview=html_note,
            )
        except Exception as e:
            return SyncResult(
                success=False,
                action="error",
                message=f"Failed to create entry: {e}",
            )


async def get_today_status(client: MCPClient) -> dict[str, Any]:
    """Get current status of today's time entry.

    Uses raw JSON methods and Ollama formatters for resilient parsing.

    Returns:
        Dict with today's time entry info and calendar events
    """
    today = date.today()
    today_str = today.isoformat()

    result = {
        "date": today_str,
        "time_entry": None,
        "events": [],
        "events_count": 0,
        "calendar_error": None,
        "entries_error": None,
        "raw_events": None,
        "raw_entries": None,
    }

    # Get calendar events - try raw method first for flexibility
    try:
        raw_events = await client.get_events_today_raw()
        result["raw_events"] = raw_events

        # Check for error response
        if isinstance(raw_events, dict) and "error" in raw_events:
            result["calendar_error"] = raw_events["error"]
        else:
            # Extract events from various response formats
            events_list = []
            if isinstance(raw_events, dict):
                events_list = raw_events.get("events", [])
            elif isinstance(raw_events, list):
                events_list = raw_events

            # Format events for display
            formatted_events = []
            for e in events_list:
                if isinstance(e, dict):
                    formatted_events.append({
                        "summary": e.get("summary", "Unknown"),
                        "start": _extract_time(e.get("start", "")),
                        "end": _extract_time(e.get("end", "")),
                    })

            result["events"] = formatted_events
            result["events_count"] = len(formatted_events)

    except Exception as e:
        result["calendar_error"] = str(e)
        # Fallback to old method
        try:
            events = await client.get_events_today()
            result["calendar_error"] = None  # Clear error if fallback works
            result["events"] = [
                {
                    "summary": e.summary,
                    "start": e.start.strftime("%H:%M"),
                    "end": e.end.strftime("%H:%M"),
                }
                for e in events
            ]
            result["events_count"] = len(events)
        except Exception:
            pass

    # Get time entry - try raw method first
    try:
        raw_entries = await client.get_time_entries_raw(after=today, before=today)
        result["raw_entries"] = raw_entries

        # Extract entries from various response formats
        entries_list = []
        if isinstance(raw_entries, dict):
            entries_list = raw_entries.get("entries", raw_entries.get("data", []))
            if not isinstance(entries_list, list):
                entries_list = [raw_entries] if raw_entries.get("id") else []
        elif isinstance(raw_entries, list):
            entries_list = raw_entries

        # Find today's entry
        for entry in entries_list:
            if isinstance(entry, dict):
                entry_date = entry.get("date")
                # Handle various date formats including None
                if entry_date is None or entry_date == today_str or str(entry_date).startswith(today_str):
                    full_id = entry.get("id", "")

                    # Try to get full entry details for accurate hours
                    # Parse numeric ID from report format: "time-entry-report-...-137120906-hash"
                    numeric_id = None
                    parts = full_id.split("-")
                    for part in reversed(parts):
                        if part.isdigit():
                            numeric_id = part
                            break

                    time_minutes = 0
                    hours = 0
                    note = ""

                    if numeric_id:
                        try:
                            import json
                            full_entry = await client._call_tool(
                                client._productive_session,
                                "get_time_entry",
                                {"entry_id": numeric_id},
                            )
                            if full_entry:
                                data = json.loads(full_entry)
                                hours = data.get("hours", 0)
                                time_minutes = int(float(hours) * 60) if hours else 0
                                note = data.get("note", "")
                                full_id = data.get("id", full_id)
                        except Exception:
                            # Fallback to report data
                            hours = entry.get("hours", 0)
                            time_minutes = int(float(hours) * 60) if hours else 0
                            note = entry.get("note") or entry.get("description", "")
                    else:
                        hours = entry.get("hours", 0)
                        time_minutes = int(float(hours) * 60) if hours else 0
                        note = entry.get("note") or entry.get("description", "")

                    result["time_entry"] = {
                        "id": full_id,
                        "time_minutes": time_minutes,
                        "time_hours": hours if hours else (time_minutes / 60 if time_minutes else 0),
                        "note": note,
                    }
                    break

    except Exception:
        # Fallback to old method
        try:
            entries = await client.get_time_entries(after=today, before=today)
            today_entries = [e for e in entries if e.date == today_str]
            if today_entries:
                entry = today_entries[0]
                result["time_entry"] = {
                    "id": entry.id,
                    "time_minutes": entry.time,
                    "time_hours": entry.time / 60,
                    "note": entry.note,
                }
        except Exception:
            pass

    return result


async def _get_today_entry_with_note(client: MCPClient) -> tuple[str | None, str | None]:
    """Get today's entry ID and current note content.

    Fetches the full entry details to get the actual note.

    Returns:
        Tuple of (entry_id, existing_note) or (None, None) if no entry exists
    """
    import json

    today = date.today()
    today_str = today.isoformat()

    try:
        raw_entries = await client.get_time_entries_raw(after=today, before=today)

        entries_list = []
        if isinstance(raw_entries, dict):
            entries_list = raw_entries.get("entries", raw_entries.get("data", []))
            if not isinstance(entries_list, list):
                entries_list = [raw_entries] if raw_entries.get("id") else []
        elif isinstance(raw_entries, list):
            entries_list = raw_entries

        # Find today's entry
        for entry in entries_list:
            if isinstance(entry, dict):
                entry_date = entry.get("date")
                if entry_date is None or entry_date == today_str or str(entry_date).startswith(today_str):
                    # Parse numeric ID from report format
                    full_id = entry.get("id", "")
                    entry_id = _extract_numeric_id(full_id)

                    # Fetch full entry details to get actual note
                    if entry_id:
                        try:
                            full_entry = await client._call_tool(
                                client._productive_session,
                                "get_time_entry",
                                {"entry_id": entry_id},
                            )
                            if full_entry:
                                data = json.loads(full_entry)
                                existing_note = data.get("note", "")
                                return entry_id, existing_note
                        except Exception:
                            pass

                    # Fallback to report note (may be empty)
                    existing_note = entry.get("note") or entry.get("description", "")
                    return entry_id, existing_note

    except Exception:
        # Fallback to old method
        try:
            entries = await client.get_time_entries(after=today, before=today)
            today_entries = [e for e in entries if e.date == today_str]
            if today_entries:
                return today_entries[0].id, today_entries[0].note
        except Exception:
            pass

    return None, None


async def add_item_to_today(client: MCPClient, item: str) -> SyncResult:
    """Add an item to today's time entry note (appends to existing).

    Fetches the full entry details to preserve existing note content.

    Args:
        client: Connected MCP client
        item: Item text to add

    Returns:
        SyncResult with operation details
    """
    today = date.today()
    today_str = today.isoformat()

    # Get existing entry with full note content
    entry_id, existing_note = await _get_today_entry_with_note(client)

    if not entry_id:
        # Create new entry with just this item
        html_note = f"<ul>\n<li>{item}</li>\n</ul>"
        try:
            result = await client.create_time_entry(
                date_str=today_str,
                time_minutes=480,
                note=html_note,
            )

            # Check for service selection needed
            if isinstance(result, dict) and result.get("needs_service_selection"):
                services = result.get("available_services", [])
                service_list = "\n".join(
                    f"  - {s.get('service_name')} (ID: {s.get('service_id')}, used {s.get('count')}x)"
                    for s in services[:5]
                )
                return SyncResult(
                    success=False,
                    action="error",
                    message=f"Multiple services found. Set PRODUCTIVE_SERVICE_ID to one of:\n{service_list}",
                )

            # Check for error in result
            if isinstance(result, dict) and (result.get("error") or ("raw" in result and "Error" in str(result.get("raw", "")))):
                return SyncResult(
                    success=False,
                    action="error",
                    message=str(result.get("raw", "Unknown error")),
                )

            # Extract entry ID
            new_entry_id = result.get("id") if isinstance(result, dict) else None

            return SyncResult(
                success=True,
                action="created",
                message=f"Created time entry with item: {item}",
                time_entry_id=new_entry_id,
                note_preview=html_note,
            )
        except Exception as e:
            return SyncResult(
                success=False,
                action="error",
                message=f"Failed to create entry: {e}",
            )

    # Update existing entry - append to existing note
    existing_note = existing_note or ""

    # Add item to existing list
    if "</ul>" in existing_note:
        new_note = existing_note.replace("</ul>", f"<li>{item}</li>\n</ul>")
    elif existing_note.strip():
        new_note = f"{existing_note}\n<ul>\n<li>{item}</li>\n</ul>"
    else:
        new_note = f"<ul>\n<li>{item}</li>\n</ul>"

    try:
        await client.update_time_entry(entry_id, note=new_note)
        return SyncResult(
            success=True,
            action="updated",
            message=f"Added item to time entry: {item}",
            time_entry_id=entry_id,
            note_preview=new_note,
        )
    except Exception as e:
        return SyncResult(
            success=False,
            action="error",
            message=f"Failed to update entry: {e}",
        )


async def remove_item_from_today(client: MCPClient, item_pattern: str) -> SyncResult:
    """Remove an item from today's time entry note.

    Args:
        client: Connected MCP client
        item_pattern: Item text or pattern to remove (case-insensitive partial match)

    Returns:
        SyncResult with operation details
    """
    import re

    entry_id, existing_note = await _get_today_entry_with_note(client)

    if not entry_id:
        return SyncResult(
            success=False,
            action="error",
            message="No time entry found for today",
        )

    if not existing_note:
        return SyncResult(
            success=False,
            action="error",
            message="Today's entry has no note content",
        )

    # Parse items from HTML list
    items = _parse_note_items(existing_note)
    pattern_lower = item_pattern.lower()

    # Find and remove matching items
    removed = []
    remaining = []
    for item in items:
        if pattern_lower in item.lower():
            removed.append(item)
        else:
            remaining.append(item)

    if not removed:
        return SyncResult(
            success=False,
            action="error",
            message=f"No items matching '{item_pattern}' found",
        )

    # Build new note
    new_note = _build_note_html(remaining)

    try:
        await client.update_time_entry(entry_id, note=new_note)
        removed_str = ", ".join(removed[:3])
        if len(removed) > 3:
            removed_str += f" (+{len(removed) - 3} more)"
        return SyncResult(
            success=True,
            action="updated",
            message=f"Removed {len(removed)} item(s): {removed_str}",
            time_entry_id=entry_id,
            note_preview=new_note,
        )
    except Exception as e:
        return SyncResult(
            success=False,
            action="error",
            message=f"Failed to update entry: {e}",
        )


async def substitute_item_today(
    client: MCPClient, old_pattern: str, new_item: str
) -> SyncResult:
    """Replace an item in today's time entry note.

    Args:
        client: Connected MCP client
        old_pattern: Item text or pattern to replace (case-insensitive partial match)
        new_item: New item text to replace with

    Returns:
        SyncResult with operation details
    """
    entry_id, existing_note = await _get_today_entry_with_note(client)

    if not entry_id:
        return SyncResult(
            success=False,
            action="error",
            message="No time entry found for today",
        )

    if not existing_note:
        return SyncResult(
            success=False,
            action="error",
            message="Today's entry has no note content",
        )

    # Parse items from HTML list
    items = _parse_note_items(existing_note)
    pattern_lower = old_pattern.lower()

    # Find and replace matching items
    replaced_count = 0
    new_items = []
    for item in items:
        if pattern_lower in item.lower():
            new_items.append(new_item)
            replaced_count += 1
        else:
            new_items.append(item)

    if replaced_count == 0:
        return SyncResult(
            success=False,
            action="error",
            message=f"No items matching '{old_pattern}' found",
        )

    # Build new note
    new_note = _build_note_html(new_items)

    try:
        await client.update_time_entry(entry_id, note=new_note)
        return SyncResult(
            success=True,
            action="updated",
            message=f"Replaced {replaced_count} item(s) with: {new_item}",
            time_entry_id=entry_id,
            note_preview=new_note,
        )
    except Exception as e:
        return SyncResult(
            success=False,
            action="error",
            message=f"Failed to update entry: {e}",
        )


async def clear_note_today(client: MCPClient) -> SyncResult:
    """Clear today's time entry note entirely.

    Args:
        client: Connected MCP client

    Returns:
        SyncResult with operation details
    """
    entry_id, existing_note = await _get_today_entry_with_note(client)

    if not entry_id:
        return SyncResult(
            success=False,
            action="error",
            message="No time entry found for today",
        )

    try:
        await client.update_time_entry(entry_id, note="")
        return SyncResult(
            success=True,
            action="updated",
            message="Cleared time entry note",
            time_entry_id=entry_id,
            note_preview="",
        )
    except Exception as e:
        return SyncResult(
            success=False,
            action="error",
            message=f"Failed to update entry: {e}",
        )


async def set_note_today(client: MCPClient, note: str) -> SyncResult:
    """Set today's time entry note (replaces entirely).

    Args:
        client: Connected MCP client
        note: Full note content (can be plain text or HTML)

    Returns:
        SyncResult with operation details
    """
    today = date.today()
    today_str = today.isoformat()

    entry_id, _ = await _get_today_entry_with_note(client)

    # Format as HTML list if plain text
    if note and not note.strip().startswith("<"):
        # Split by newlines and create list items
        lines = [line.strip() for line in note.split("\n") if line.strip()]
        html_note = "<ul>\n" + "\n".join(f"<li>{line}</li>" for line in lines) + "\n</ul>"
    else:
        html_note = note

    if not entry_id:
        # Create new entry
        try:
            result = await client.create_time_entry(
                date_str=today_str,
                time_minutes=480,
                note=html_note,
            )

            if isinstance(result, dict) and result.get("needs_service_selection"):
                services = result.get("available_services", [])
                service_list = "\n".join(
                    f"  - {s.get('service_name')} (ID: {s.get('service_id')}, used {s.get('count')}x)"
                    for s in services[:5]
                )
                return SyncResult(
                    success=False,
                    action="error",
                    message=f"Multiple services found. Set PRODUCTIVE_SERVICE_ID to one of:\n{service_list}",
                )

            if isinstance(result, dict) and (result.get("error") or ("raw" in result and "Error" in str(result.get("raw", "")))):
                return SyncResult(
                    success=False,
                    action="error",
                    message=str(result.get("raw", "Unknown error")),
                )

            new_entry_id = result.get("id") if isinstance(result, dict) else None

            return SyncResult(
                success=True,
                action="created",
                message="Created time entry with note",
                time_entry_id=new_entry_id,
                note_preview=html_note,
            )
        except Exception as e:
            return SyncResult(
                success=False,
                action="error",
                message=f"Failed to create entry: {e}",
            )

    try:
        await client.update_time_entry(entry_id, note=html_note)
        return SyncResult(
            success=True,
            action="updated",
            message="Set time entry note",
            time_entry_id=entry_id,
            note_preview=html_note,
        )
    except Exception as e:
        return SyncResult(
            success=False,
            action="error",
            message=f"Failed to update entry: {e}",
        )


def _parse_note_items(note: str) -> list[str]:
    """Parse list items from an HTML note.

    Args:
        note: HTML note content

    Returns:
        List of item texts (without HTML tags)
    """
    import re

    items = []
    # Match <li>content</li> patterns
    pattern = r"<li>(.*?)</li>"
    matches = re.findall(pattern, note, re.IGNORECASE | re.DOTALL)
    for match in matches:
        # Strip nested tags like <em>
        clean = re.sub(r"<[^>]+>", "", match)
        clean = clean.strip()
        if clean:
            items.append(clean)
    return items


def _build_note_html(items: list[str]) -> str:
    """Build HTML note from list of items.

    Args:
        items: List of item texts

    Returns:
        HTML formatted note
    """
    if not items:
        return ""
    return "<ul>\n" + "\n".join(f"<li>{item}</li>" for item in items) + "\n</ul>"


def _extract_time(dt_string: str) -> str:
    """Extract HH:MM time from an ISO datetime string."""
    if not dt_string:
        return ""
    # Handle ISO format: 2026-03-11T09:00:00 or 2026-03-11T09:00:00+00:00
    if "T" in dt_string:
        time_part = dt_string.split("T")[1]
        return time_part[:5]  # HH:MM
    # Already just a time
    if ":" in dt_string:
        return dt_string[:5]
    return dt_string


def _extract_numeric_id(full_id: str) -> str:
    """Extract numeric entry ID from report format.

    Report IDs look like: "time-entry-report-time_entry-137120906-e9036a78aaab..."
    The actual entry ID is the numeric part: "137120906"
    """
    if not full_id:
        return ""
    # If it's already just a number, return it
    if full_id.isdigit():
        return full_id
    # Parse from report format
    parts = full_id.split("-")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return full_id  # Fallback to original


def _filter_events(events: list[CalendarEvent]) -> list[CalendarEvent]:
    """Filter out placeholder events from CalendarEvent objects."""
    skip_keywords = {"busy", "private", "blocked", "focus time", "lunch"}
    return [
        e
        for e in events
        if e.summary.lower() not in skip_keywords
        and not e.summary.lower().startswith("busy")
    ]


def _filter_events_raw(events: list[dict]) -> list[dict]:
    """Filter out placeholder events from raw dict events."""
    skip_keywords = {"busy", "private", "blocked", "focus time", "lunch"}
    filtered = []
    for e in events:
        if isinstance(e, dict):
            summary = e.get("summary", "").lower()
            if summary not in skip_keywords and not summary.startswith("busy"):
                filtered.append(e)
    return filtered
