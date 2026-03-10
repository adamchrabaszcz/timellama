"""Core sync logic for calendar to Productive time entries."""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from timellama.mcp_client import CalendarEvent, MCPClient, TimeEntry
from timellama.ollama_client import format_events_to_html, generate_suggestions


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

    Args:
        client: Connected MCP client
        dry_run: If True, don't actually create/update entries
        include_suggestions: If True, include LLM-generated suggestions

    Returns:
        SyncResult with operation details
    """
    today = date.today()
    today_str = today.isoformat()

    # Get today's calendar events
    try:
        events = await client.get_events_today()
    except Exception as e:
        return SyncResult(
            success=False,
            action="error",
            message=f"Failed to fetch calendar events: {e}",
        )

    # Filter out "busy" events
    filtered_events = _filter_events(events)

    # Format events to HTML
    html_note = format_events_to_html(filtered_events)

    # Optionally add suggestions
    if include_suggestions and filtered_events:
        suggestions = generate_suggestions(filtered_events, html_note)
        if suggestions:
            suggestion_items = "\n".join([f"<li>{s}</li>" for s in suggestions])
            html_note = html_note.replace(
                "</ul>", f"<li><em>Other: {', '.join(suggestions)}</em></li>\n</ul>"
            )

    # Check for existing time entry today
    try:
        entries = await client.get_time_entries(after=today, before=today)
        today_entries = [e for e in entries if e.date == today_str]
    except Exception as e:
        return SyncResult(
            success=False,
            action="error",
            message=f"Failed to fetch existing entries: {e}",
        )

    if dry_run:
        action = "would update" if today_entries else "would create"
        return SyncResult(
            success=True,
            action="skipped",
            message=f"Dry run: {action} time entry with {len(filtered_events)} events",
            events_count=len(filtered_events),
            note_preview=html_note,
        )

    # Create or update entry
    if today_entries:
        # Update existing entry
        entry = today_entries[0]
        try:
            result = await client.update_time_entry(entry.id, note=html_note)
            return SyncResult(
                success=True,
                action="updated",
                message=f"Updated time entry with {len(filtered_events)} events",
                time_entry_id=entry.id,
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
            entry_id = result.get("id") if isinstance(result, dict) else None
            return SyncResult(
                success=True,
                action="created",
                message=f"Created time entry with {len(filtered_events)} events",
                time_entry_id=entry_id,
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
    }

    # Get calendar events
    try:
        events = await client.get_events_today()
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

    # Get time entry
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


async def add_item_to_today(client: MCPClient, item: str) -> SyncResult:
    """Add an item to today's time entry note.

    Args:
        client: Connected MCP client
        item: Item text to add

    Returns:
        SyncResult with operation details
    """
    today = date.today()
    today_str = today.isoformat()

    # Get existing entry
    try:
        entries = await client.get_time_entries(after=today, before=today)
        today_entries = [e for e in entries if e.date == today_str]
    except Exception as e:
        return SyncResult(
            success=False,
            action="error",
            message=f"Failed to fetch existing entries: {e}",
        )

    if not today_entries:
        # Create new entry with just this item
        html_note = f"<ul>\n<li>{item}</li>\n</ul>"
        try:
            result = await client.create_time_entry(
                date_str=today_str,
                time_minutes=480,
                note=html_note,
            )
            entry_id = result.get("id") if isinstance(result, dict) else None
            return SyncResult(
                success=True,
                action="created",
                message=f"Created time entry with item: {item}",
                time_entry_id=entry_id,
                note_preview=html_note,
            )
        except Exception as e:
            return SyncResult(
                success=False,
                action="error",
                message=f"Failed to create entry: {e}",
            )

    # Update existing entry
    entry = today_entries[0]
    existing_note = entry.note or "<ul>\n</ul>"

    # Add item to existing list
    if "</ul>" in existing_note:
        new_note = existing_note.replace("</ul>", f"<li>{item}</li>\n</ul>")
    else:
        new_note = f"{existing_note}\n<ul>\n<li>{item}</li>\n</ul>"

    try:
        await client.update_time_entry(entry.id, note=new_note)
        return SyncResult(
            success=True,
            action="updated",
            message=f"Added item to time entry: {item}",
            time_entry_id=entry.id,
            note_preview=new_note,
        )
    except Exception as e:
        return SyncResult(
            success=False,
            action="error",
            message=f"Failed to update entry: {e}",
        )


def _filter_events(events: list[CalendarEvent]) -> list[CalendarEvent]:
    """Filter out placeholder events."""
    skip_keywords = {"busy", "private", "blocked", "focus time", "lunch"}
    return [
        e
        for e in events
        if e.summary.lower() not in skip_keywords
        and not e.summary.lower().startswith("busy")
    ]
