"""Tests for sync logic."""

from datetime import datetime

import pytest

from timellama.mcp_client import CalendarEvent
from timellama.sync import _filter_events


class TestFilterEvents:
    """Tests for event filtering."""

    def test_keeps_regular_events(self):
        """Regular meetings are kept."""
        events = [
            CalendarEvent(
                summary="Team Standup",
                start=datetime(2026, 1, 15, 9, 0),
                end=datetime(2026, 1, 15, 9, 30),
            ),
            CalendarEvent(
                summary="1:1 with Manager",
                start=datetime(2026, 1, 15, 10, 0),
                end=datetime(2026, 1, 15, 10, 30),
            ),
        ]
        result = _filter_events(events)
        assert len(result) == 2

    def test_filters_busy(self):
        """Busy events are filtered."""
        events = [
            CalendarEvent(
                summary="Busy",
                start=datetime(2026, 1, 15, 9, 0),
                end=datetime(2026, 1, 15, 10, 0),
            ),
            CalendarEvent(
                summary="busy - personal",
                start=datetime(2026, 1, 15, 11, 0),
                end=datetime(2026, 1, 15, 12, 0),
            ),
        ]
        result = _filter_events(events)
        assert len(result) == 0

    def test_filters_private(self):
        """Private events are filtered."""
        events = [
            CalendarEvent(
                summary="Private",
                start=datetime(2026, 1, 15, 9, 0),
                end=datetime(2026, 1, 15, 10, 0),
            ),
        ]
        result = _filter_events(events)
        assert len(result) == 0

    def test_filters_focus_time(self):
        """Focus time events are filtered."""
        events = [
            CalendarEvent(
                summary="Focus Time",
                start=datetime(2026, 1, 15, 9, 0),
                end=datetime(2026, 1, 15, 12, 0),
            ),
        ]
        result = _filter_events(events)
        assert len(result) == 0

    def test_filters_lunch(self):
        """Lunch events are filtered."""
        events = [
            CalendarEvent(
                summary="Lunch",
                start=datetime(2026, 1, 15, 12, 0),
                end=datetime(2026, 1, 15, 13, 0),
            ),
        ]
        result = _filter_events(events)
        assert len(result) == 0

    def test_mixed_events(self):
        """Mix of events filters correctly."""
        events = [
            CalendarEvent(
                summary="Team Standup",
                start=datetime(2026, 1, 15, 9, 0),
                end=datetime(2026, 1, 15, 9, 30),
            ),
            CalendarEvent(
                summary="Busy",
                start=datetime(2026, 1, 15, 10, 0),
                end=datetime(2026, 1, 15, 11, 0),
            ),
            CalendarEvent(
                summary="Code Review",
                start=datetime(2026, 1, 15, 14, 0),
                end=datetime(2026, 1, 15, 15, 0),
            ),
            CalendarEvent(
                summary="Focus Time",
                start=datetime(2026, 1, 15, 15, 0),
                end=datetime(2026, 1, 15, 17, 0),
            ),
        ]
        result = _filter_events(events)
        assert len(result) == 2
        assert result[0].summary == "Team Standup"
        assert result[1].summary == "Code Review"
