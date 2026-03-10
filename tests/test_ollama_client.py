"""Tests for Ollama client."""

from datetime import datetime

import pytest

from timellama.mcp_client import CalendarEvent
from timellama.ollama_client import _format_events_simple, parse_chat_input


class TestFormatEventsSimple:
    """Tests for simple HTML formatting without LLM."""

    def test_empty_events(self):
        """Empty list returns no meetings message."""
        result = _format_events_simple([])
        assert "<li>No meetings today</li>" in result

    def test_single_event(self):
        """Single event is formatted correctly."""
        events = [
            CalendarEvent(
                summary="Team Standup",
                start=datetime(2026, 1, 15, 9, 0),
                end=datetime(2026, 1, 15, 9, 30),
            )
        ]
        result = _format_events_simple(events)
        assert "<li>Team Standup</li>" in result
        assert "<ul>" in result
        assert "</ul>" in result

    def test_multiple_events(self):
        """Multiple events are all included."""
        events = [
            CalendarEvent(
                summary="Standup",
                start=datetime(2026, 1, 15, 9, 0),
                end=datetime(2026, 1, 15, 9, 30),
            ),
            CalendarEvent(
                summary="Code Review",
                start=datetime(2026, 1, 15, 14, 0),
                end=datetime(2026, 1, 15, 15, 0),
            ),
        ]
        result = _format_events_simple(events)
        assert "<li>Standup</li>" in result
        assert "<li>Code Review</li>" in result

    def test_filters_busy_events(self):
        """Busy/private events are filtered out."""
        events = [
            CalendarEvent(
                summary="Busy",
                start=datetime(2026, 1, 15, 9, 0),
                end=datetime(2026, 1, 15, 10, 0),
            ),
            CalendarEvent(
                summary="Private",
                start=datetime(2026, 1, 15, 10, 0),
                end=datetime(2026, 1, 15, 11, 0),
            ),
            CalendarEvent(
                summary="Real Meeting",
                start=datetime(2026, 1, 15, 14, 0),
                end=datetime(2026, 1, 15, 15, 0),
            ),
        ]
        result = _format_events_simple(events)
        assert "Busy" not in result
        assert "Private" not in result
        assert "<li>Real Meeting</li>" in result


class TestParseChatInput:
    """Tests for chat input parsing."""

    def test_quit_commands(self):
        """Quit commands are recognized."""
        for cmd in ["quit", "exit", "q"]:
            result = parse_chat_input(cmd)
            assert result["intent"] == "quit"

    def test_sync_command(self):
        """Sync command is recognized."""
        result = parse_chat_input("sync")
        assert result["intent"] == "sync"

    def test_show_command(self):
        """Show command is recognized."""
        for cmd in ["show", "status", "current"]:
            result = parse_chat_input(cmd)
            assert result["intent"] == "show"

    def test_add_command(self):
        """Add command extracts item."""
        result = parse_chat_input("add Code review")
        assert result["intent"] == "add"
        assert result["data"] == "Code review"

    def test_remove_command(self):
        """Remove command extracts item."""
        result = parse_chat_input("remove Meeting notes")
        assert result["intent"] == "remove"
        assert result["data"] == "Meeting notes"

    def test_case_insensitive(self):
        """Commands are case insensitive."""
        result = parse_chat_input("SYNC")
        assert result["intent"] == "sync"

        result = parse_chat_input("QUIT")
        assert result["intent"] == "quit"
