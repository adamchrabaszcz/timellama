"""MCP client wrapper for spawning and communicating with bundled MCP servers."""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class TimeEntry:
    """Represents a Productive time entry."""

    id: str
    date: str
    time: int  # minutes
    note: str | None
    person_id: str
    service_id: str | None


@dataclass
class CalendarEvent:
    """Represents a calendar event."""

    summary: str
    start: datetime
    end: datetime
    description: str | None = None
    location: str | None = None


class MCPClient:
    """Client for interacting with bundled MCP servers."""

    def __init__(self):
        self._productive_session: ClientSession | None = None
        self._calendar_session: ClientSession | None = None
        self._productive_context = None
        self._calendar_context = None

    @asynccontextmanager
    async def connect(self):
        """Connect to both MCP servers."""
        # Build server parameters using sys.executable -m to ensure we use
        # the same Python environment where timellama is installed (works with pipx)
        productive_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "productive_time_mcp"],
            env={
                **os.environ,
                "PRODUCTIVE_API_TOKEN": os.environ.get("PRODUCTIVE_API_TOKEN", ""),
                "PRODUCTIVE_ORG_ID": os.environ.get("PRODUCTIVE_ORG_ID", ""),
                "PRODUCTIVE_USER_ID": os.environ.get("PRODUCTIVE_USER_ID", ""),
                "PRODUCTIVE_BILLING_CUTOFF_DAY": os.environ.get(
                    "PRODUCTIVE_BILLING_CUTOFF_DAY", "10"
                ),
            },
        )

        calendar_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ics_calendar_mcp"],
            env={
                **os.environ,
                "ICS_CALENDAR_URL": os.environ.get("ICS_CALENDAR_URL", ""),
            },
        )

        # Connect to both servers
        async with stdio_client(productive_params) as (prod_read, prod_write):
            async with stdio_client(calendar_params) as (cal_read, cal_write):
                async with ClientSession(prod_read, prod_write) as prod_session:
                    async with ClientSession(cal_read, cal_write) as cal_session:
                        await prod_session.initialize()
                        await cal_session.initialize()
                        self._productive_session = prod_session
                        self._calendar_session = cal_session
                        try:
                            yield self
                        finally:
                            self._productive_session = None
                            self._calendar_session = None

    async def _call_tool(
        self, session: ClientSession, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """Call a tool on an MCP server."""
        result = await session.call_tool(tool_name, arguments)
        if result.content:
            # Return the text content
            for content in result.content:
                if hasattr(content, "text"):
                    return content.text
        return None

    async def _call_tool_raw(
        self, session: ClientSession, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool and return raw JSON response as dict.

        This method is preferred for the new architecture where Ollama
        handles the formatting instead of rigid dataclass parsing.
        """
        import json
        result = await self._call_tool(session, tool_name, arguments)
        if result:
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"raw": result}
        return {}

    # Calendar methods
    async def get_events_today(self) -> list[CalendarEvent]:
        """Get today's calendar events."""
        if not self._calendar_session:
            raise RuntimeError("Not connected to calendar server")

        result = await self._call_tool(self._calendar_session, "get_events_today", {})
        return self._parse_events(result)

    async def get_events_today_raw(self) -> dict[str, Any]:
        """Get today's calendar events as raw JSON."""
        if not self._calendar_session:
            raise RuntimeError("Not connected to calendar server")

        return await self._call_tool_raw(self._calendar_session, "get_events_today", {})

    async def get_events_range(
        self, start_date: date, end_date: date
    ) -> list[CalendarEvent]:
        """Get calendar events in a date range."""
        if not self._calendar_session:
            raise RuntimeError("Not connected to calendar server")

        result = await self._call_tool(
            self._calendar_session,
            "get_events_range",
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
        return self._parse_events(result)

    async def get_events_range_raw(
        self, start_date: date, end_date: date
    ) -> dict[str, Any]:
        """Get calendar events in a date range as raw JSON."""
        if not self._calendar_session:
            raise RuntimeError("Not connected to calendar server")

        return await self._call_tool_raw(
            self._calendar_session,
            "get_events_range",
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

    def _parse_events(self, result: str | None) -> list[CalendarEvent]:
        """Parse calendar events from MCP response."""
        if not result:
            return []

        events = []
        import json

        try:
            data = json.loads(result)
            # Handle both list format and dict with 'events' key
            event_list = data.get("events", data) if isinstance(data, dict) else data
            if isinstance(event_list, list):
                for event in event_list:
                    events.append(
                        CalendarEvent(
                            summary=event.get("summary", "Unknown"),
                            start=datetime.fromisoformat(event.get("start", "")),
                            end=datetime.fromisoformat(event.get("end", "")),
                            description=event.get("description"),
                            location=event.get("location"),
                        )
                    )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Try parsing as text format
            pass

        return events

    # Productive time methods
    async def get_my_hours(
        self,
        after: date | None = None,
        before: date | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        """Get current user's time entries."""
        if not self._productive_session:
            raise RuntimeError("Not connected to productive server")

        args: dict[str, Any] = {}
        if after:
            args["after"] = after.isoformat()
        if before:
            args["before"] = before.isoformat()
        if period:
            args["period"] = period

        result = await self._call_tool(self._productive_session, "get_my_hours", args)
        import json

        try:
            return json.loads(result) if result else {}
        except json.JSONDecodeError:
            return {"raw": result}

    async def get_employee_hours(
        self,
        name: str,
        after: date | None = None,
        before: date | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        """Get an employee's time entries by name."""
        if not self._productive_session:
            raise RuntimeError("Not connected to productive server")

        args: dict[str, Any] = {"name": name}
        if after:
            args["after"] = after.isoformat()
        if before:
            args["before"] = before.isoformat()
        if period:
            args["period"] = period

        result = await self._call_tool(
            self._productive_session, "get_employee_hours", args
        )
        import json

        try:
            return json.loads(result) if result else {}
        except json.JSONDecodeError:
            return {"raw": result}

    async def get_time_entries(
        self,
        person_id: str | None = None,
        after: date | None = None,
        before: date | None = None,
    ) -> list[TimeEntry]:
        """Get time entries with optional filters."""
        if not self._productive_session:
            raise RuntimeError("Not connected to productive server")

        args: dict[str, Any] = {}
        if person_id:
            args["person_id"] = person_id
        if after:
            args["after"] = after.isoformat()
        if before:
            args["before"] = before.isoformat()

        result = await self._call_tool(
            self._productive_session, "get_time_entries", args
        )
        return self._parse_time_entries(result)

    async def get_time_entries_raw(
        self,
        person_id: str | None = None,
        after: date | None = None,
        before: date | None = None,
    ) -> dict[str, Any]:
        """Get time entries as raw JSON with optional filters."""
        if not self._productive_session:
            raise RuntimeError("Not connected to productive server")

        args: dict[str, Any] = {}
        if person_id:
            args["person_id"] = person_id
        if after:
            args["after"] = after.isoformat()
        if before:
            args["before"] = before.isoformat()

        return await self._call_tool_raw(
            self._productive_session, "get_time_entries", args
        )

    def _parse_time_entries(self, result: str | None) -> list[TimeEntry]:
        """Parse time entries from MCP response."""
        if not result:
            return []

        entries = []
        import json

        try:
            data = json.loads(result)
            # Handle both list format and dict with 'entries' key
            entry_list = data.get("entries", data) if isinstance(data, dict) else data
            if isinstance(entry_list, list):
                for entry in entry_list:
                    entries.append(
                        TimeEntry(
                            id=entry.get("id", ""),
                            date=entry.get("date", ""),
                            time=entry.get("time", 0),
                            note=entry.get("note"),
                            person_id=entry.get("person_id", ""),
                            service_id=entry.get("service_id"),
                        )
                    )
        except (json.JSONDecodeError, KeyError):
            pass

        return entries

    async def get_recent_services(self, days: int = 14) -> list[dict[str, Any]]:
        """Get services used in recent time entries.

        Returns list of dicts with service_id, service_name, count.
        """
        import json
        from datetime import timedelta

        if not self._productive_session:
            return []

        today = date.today()
        start_date = today - timedelta(days=days)

        try:
            # Get recent entry IDs from reports
            raw = await self._call_tool_raw(
                self._productive_session,
                "get_time_entries",
                {"after": start_date.isoformat(), "before": today.isoformat()},
            )

            entries = []
            if isinstance(raw, dict):
                entries = raw.get("entries", raw.get("data", []))
            elif isinstance(raw, list):
                entries = raw

            # Fetch full details for each entry to get service info
            service_counts: dict[str, dict] = {}
            for entry in entries[:20]:  # Limit to 20 to avoid too many API calls
                if isinstance(entry, dict):
                    full_id = entry.get("id", "")
                    # Parse the numeric ID from report format
                    # e.g., "time-entry-report-time_entry-137120906-hash" -> "137120906"
                    parts = full_id.split("-")
                    numeric_id = None
                    for part in reversed(parts):
                        if part.isdigit():
                            numeric_id = part
                            break

                    if numeric_id:
                        try:
                            result = await self._call_tool(
                                self._productive_session,
                                "get_time_entry",
                                {"entry_id": numeric_id},
                            )
                            if result:
                                data = json.loads(result)
                                service = data.get("service", {})
                                sid = service.get("id")
                                sname = service.get("name", "")
                                if sid:
                                    if sid not in service_counts:
                                        service_counts[sid] = {
                                            "service_id": sid,
                                            "service_name": sname,
                                            "count": 0,
                                        }
                                    service_counts[sid]["count"] += 1
                        except Exception:
                            continue

            # Sort by count descending
            return sorted(service_counts.values(), key=lambda x: x["count"], reverse=True)

        except Exception:
            return []

    async def create_time_entry(
        self,
        date_str: str,
        time_minutes: int,
        note: str,
        service_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new time entry."""
        if not self._productive_session:
            raise RuntimeError("Not connected to productive server")

        # Convert minutes to hours for the MCP server
        hours = time_minutes / 60

        # Get default service_id from environment if not provided
        if not service_id:
            service_id = os.environ.get("PRODUCTIVE_SERVICE_ID")

        # If still no service_id, try to detect from recent entries
        if not service_id:
            recent = await self.get_recent_services()
            if len(recent) == 1:
                # Only one service used recently - use it automatically
                service_id = recent[0]["service_id"]
            elif len(recent) > 1:
                # Multiple services - return info for user to select
                return {
                    "needs_service_selection": True,
                    "available_services": recent,
                    "error": "Multiple services found in recent entries.",
                }

        if not service_id:
            return {
                "error": True,
                "raw": (
                    "Error: No service_id found.\n"
                    "Set PRODUCTIVE_SERVICE_ID in your environment, or\n"
                    "create at least one time entry in Productive first."
                ),
            }

        args: dict[str, Any] = {
            "date": date_str,
            "hours": hours,
            "note": note,
            "service_id": service_id,
        }

        result = await self._call_tool(
            self._productive_session, "create_time_entry", args
        )
        import json

        try:
            return json.loads(result) if result else {}
        except json.JSONDecodeError:
            return {"raw": result}

    async def update_time_entry(
        self,
        entry_id: str,
        time_minutes: int | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing time entry."""
        if not self._productive_session:
            raise RuntimeError("Not connected to productive server")

        args: dict[str, Any] = {"entry_id": entry_id}
        if time_minutes is not None:
            # Convert minutes to hours for the MCP server
            args["hours"] = time_minutes / 60
        if note is not None:
            args["note"] = note

        result = await self._call_tool(
            self._productive_session, "update_time_entry", args
        )
        import json

        try:
            return json.loads(result) if result else {}
        except json.JSONDecodeError:
            return {"raw": result}

    async def delete_time_entry(self, entry_id: str) -> bool:
        """Delete a time entry."""
        if not self._productive_session:
            raise RuntimeError("Not connected to productive server")

        result = await self._call_tool(
            self._productive_session, "delete_time_entry", {"id": entry_id}
        )
        return result is not None

    async def get_person(self, name: str | None = None) -> dict[str, Any]:
        """Get person info by name or current user."""
        if not self._productive_session:
            raise RuntimeError("Not connected to productive server")

        args = {"name": name} if name else {}
        result = await self._call_tool(self._productive_session, "get_person", args)
        import json

        try:
            return json.loads(result) if result else {}
        except json.JSONDecodeError:
            return {"raw": result}

    async def get_time_reports(
        self,
        person_id: str | None = None,
        after: date | None = None,
        before: date | None = None,
    ) -> dict[str, Any]:
        """Get time reports with aggregated data."""
        if not self._productive_session:
            raise RuntimeError("Not connected to productive server")

        args: dict[str, Any] = {}
        if person_id:
            args["person_id"] = person_id
        if after:
            args["after"] = after.isoformat()
        if before:
            args["before"] = before.isoformat()

        result = await self._call_tool(
            self._productive_session, "get_time_reports", args
        )
        import json

        try:
            return json.loads(result) if result else {}
        except json.JSONDecodeError:
            return {"raw": result}


async def test_connection() -> tuple[bool, bool, str]:
    """Test connection to MCP servers.

    Returns:
        Tuple of (productive_ok, calendar_ok, error_message)
    """
    productive_ok = False
    calendar_ok = False
    error_msg = ""

    try:
        client = MCPClient()
        async with client.connect():
            # Test productive
            try:
                await client.get_person()
                productive_ok = True
            except Exception as e:
                error_msg += f"Productive: {e}\n"

            # Test calendar
            try:
                await client.get_events_today()
                calendar_ok = True
            except Exception as e:
                error_msg += f"Calendar: {e}\n"

    except Exception as e:
        error_msg = f"Connection error: {e}"

    return productive_ok, calendar_ok, error_msg
