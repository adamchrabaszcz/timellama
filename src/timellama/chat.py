"""Agentic chat mode for TimeLlama with Ollama tool calling."""

import asyncio
import json
from datetime import date
from typing import Any, Callable

import ollama
from ollama import ResponseError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from timellama.mcp_client import MCPClient
from timellama.ollama_client import (
    check_ollama_available,
    format_events_to_html,
    format_events_to_html_from_raw,
    format_for_display,
    get_config,
)
from timellama.sync import add_item_to_today, get_today_status, sync_today


# Tool definitions for Ollama
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_events_today",
            "description": "Get today's calendar events. Returns a list of events with summary, start time, and end time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time_entries",
            "description": "Get time entries for a date range. Returns logged hours and notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "after": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (defaults to today)",
                    },
                    "before": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (defaults to today)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_hours",
            "description": "Get the user's hours summary for a time period. Use for questions about total hours, billable time, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["current", "previous", "billing"],
                        "description": "Time period: 'current' month, 'previous' month, or 'billing' period",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_today_status",
            "description": "Get complete status for today: calendar events, logged time entry, and hours.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sync_calendar",
            "description": "Sync today's calendar events to create/update the time entry in Productive. Use when user asks to sync, refresh, or update their time log from calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_suggestions": {
                        "type": "boolean",
                        "description": "Include AI-suggested work items (default: false)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Add an item to today's time entry note. Use when user wants to add something to their log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The item text to add (e.g., 'Code review', 'PR review for #123')",
                    },
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_events_to_html",
            "description": "Format a list of events into HTML for the time entry note. Use when user wants to see how events would be formatted.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_employee_hours",
            "description": "Get hours summary and work log for any employee by name. Use when user asks about a specific person's hours, work, or time entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Employee name or partial name to search for (e.g., 'John', 'Jane Doe')",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["current", "previous", "billing"],
                        "description": "Time period: 'current' month, 'previous' month, or 'billing' period (default)",
                    },
                },
                "required": ["name"],
            },
        },
    },
]

# System prompt for the agentic chat
SYSTEM_PROMPT = """You are TimeLlama, a helpful time tracking assistant. You help users manage their Productive.io time entries and review team hours.

You have access to these tools:
- get_events_today: Fetch today's calendar events
- get_time_entries: Fetch time entries for date range
- get_my_hours: Get YOUR hours summary for a period
- get_employee_hours: Get ANY employee's hours and work log by name
- get_today_status: Get combined status (events + time entry)
- sync_calendar: Sync calendar events to time entry
- add_item: Add an item to today's time log
- format_events_to_html: Preview HTML formatting of events

When the user asks about their own status/hours, use get_my_hours or get_today_status.
When they ask about another person (e.g., "What did John work on?", "Show me Sarah's hours"), use get_employee_hours.
When they ask to sync or update, use sync_calendar.
When they want to add something to their log, use add_item.

Be concise in your responses. Format data nicely when presenting it.
When showing work logs, list the entries with dates and notes.
If a tool returns an error, explain it clearly to the user.

You can call multiple tools if needed to answer a question completely."""


async def execute_tool(client: MCPClient, tool_name: str, arguments: dict) -> dict[str, Any]:
    """Execute a tool call and return the result.

    Args:
        client: Connected MCP client
        tool_name: Name of the tool to execute
        arguments: Tool arguments

    Returns:
        Result dict with 'success' and 'data' or 'error'
    """
    try:
        if tool_name == "get_events_today":
            data = await client.get_events_today_raw()
            return {"success": True, "data": data}

        elif tool_name == "get_time_entries":
            after_str = arguments.get("after")
            before_str = arguments.get("before")
            after = date.fromisoformat(after_str) if after_str else date.today()
            before = date.fromisoformat(before_str) if before_str else date.today()
            data = await client.get_time_entries_raw(after=after, before=before)
            return {"success": True, "data": data}

        elif tool_name == "get_my_hours":
            period = arguments.get("period", "billing")
            data = await client.get_my_hours(period=period if period != "billing" else None)
            return {"success": True, "data": data}

        elif tool_name == "get_today_status":
            data = await get_today_status(client)
            return {"success": True, "data": data}

        elif tool_name == "sync_calendar":
            include_suggestions = arguments.get("include_suggestions", False)
            result = await sync_today(client, include_suggestions=include_suggestions)
            return {
                "success": result.success,
                "data": {
                    "action": result.action,
                    "message": result.message,
                    "events_count": result.events_count,
                    "note_preview": result.note_preview,
                },
            }

        elif tool_name == "add_item":
            item = arguments.get("item", "")
            if not item:
                return {"success": False, "error": "No item text provided"}
            result = await add_item_to_today(client, item)
            return {
                "success": result.success,
                "data": {"message": result.message, "action": result.action},
            }

        elif tool_name == "format_events_to_html":
            events_data = await client.get_events_today_raw()
            html = format_events_to_html_from_raw(events_data)
            return {"success": True, "data": {"html": html}}

        elif tool_name == "get_employee_hours":
            name = arguments.get("name", "")
            if not name:
                return {"success": False, "error": "Employee name is required"}
            period = arguments.get("period", "billing")
            period_arg = None if period == "billing" else period
            data = await client.get_employee_hours(name=name, period=period_arg)
            return {"success": True, "data": data}

        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def agentic_chat_loop(client: MCPClient, console: Console) -> None:
    """Run the agentic chat loop where Ollama can call MCP tools.

    Args:
        client: Connected MCP client
        console: Rich console for output
    """
    config = get_config()

    # Welcome message
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TimeLlama[/bold cyan] - Agentic Time Tracking\n\n"
            "I can help you manage your time entries using natural language.\n"
            "Try: [green]\"show my status\"[/green], [green]\"sync my calendar\"[/green], "
            "[green]\"add code review\"[/green]\n\n"
            "Type [yellow]quit[/yellow] or [yellow]exit[/yellow] to leave.",
            title="🦙 Welcome",
        )
    )

    # Initialize conversation history with system prompt
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Main loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        # Check for quit commands
        if user_input.strip().lower() in ("quit", "exit", "q", "bye"):
            console.print("[dim]Goodbye![/dim]")
            break

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        try:
            # Call Ollama with tools
            ollama_client = ollama.Client(host=config.host)
            response = ollama_client.chat(
                model=config.model,
                messages=history,
                tools=CHAT_TOOLS,
            )

            # Handle tool calls in a loop
            while response.message.tool_calls:
                # Add assistant's tool call request to history
                history.append({
                    "role": "assistant",
                    "content": response.message.content or "",
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,  # Keep as dict
                            },
                        }
                        for i, tc in enumerate(response.message.tool_calls)
                    ],
                })

                # Execute each tool call
                for i, tool_call in enumerate(response.message.tool_calls):
                    tool_name = tool_call.function.name
                    arguments = tool_call.function.arguments

                    console.print(f"[dim]→ Calling {tool_name}...[/dim]")

                    # Execute the tool
                    result = await execute_tool(client, tool_name, arguments)

                    # Add tool result to history
                    history.append({
                        "role": "tool",
                        "content": json.dumps(result, default=str),
                        "tool_call_id": f"call_{i}",
                    })

                # Continue conversation with tool results
                response = ollama_client.chat(
                    model=config.model,
                    messages=history,
                    tools=CHAT_TOOLS,
                )

            # Display final response
            final_content = response.message.content
            if final_content:
                console.print(f"\n[bold green]TimeLlama[/bold green]: {final_content}")
                history.append({"role": "assistant", "content": final_content})

        except ResponseError as e:
            console.print(f"[red]Ollama error:[/red] {e}")
            # Remove the failed user message from history
            history.pop()

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            # Remove the failed user message from history
            history.pop()


async def chat_loop(client: MCPClient, console: Console) -> None:
    """Run the interactive chat loop.

    This is the main entry point that chooses between agentic mode (if Ollama available)
    or basic mode (fallback).

    Args:
        client: Connected MCP client
        console: Rich console for output
    """
    # Check Ollama status
    ollama_ok, ollama_error = check_ollama_available()

    if ollama_ok:
        # Use agentic mode with Ollama tool calling
        await agentic_chat_loop(client, console)
    else:
        # Fallback to basic mode
        console.print(
            f"[yellow]⚠ Ollama not available:[/yellow] {ollama_error}\n"
            "[dim]Using basic mode without AI features.[/dim]\n"
        )
        await basic_chat_loop(client, console)


async def basic_chat_loop(client: MCPClient, console: Console) -> None:
    """Basic chat loop without Ollama (fallback mode).

    Args:
        client: Connected MCP client
        console: Rich console for output
    """
    # Welcome message
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TimeLlama[/bold cyan] - Basic Mode\n\n"
            "Commands: [green]sync[/green], [green]show[/green], [green]add <item>[/green], "
            "[green]hours[/green], [green]help[/green], [green]quit[/green]",
            title="🦙 Welcome",
        )
    )

    # Show current status
    await _show_status(client, console)

    # Main loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]timellama[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        input_lower = user_input.strip().lower()

        # Handle commands
        if input_lower in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        elif input_lower == "help":
            _show_help(console)

        elif input_lower in ("sync", "update", "refresh"):
            await _handle_sync(client, console)

        elif input_lower in ("show", "status", "current"):
            await _show_status(client, console)

        elif input_lower.startswith("add "):
            item = user_input[4:].strip()
            if item:
                await _handle_add(client, console, item)
            else:
                console.print("[yellow]Usage: add <item text>[/yellow]")

        elif input_lower in ("hours", "my hours"):
            await _handle_hours(client, console)

        else:
            console.print(
                f"[dim]Unknown command. Type 'help' for available commands.[/dim]"
            )


async def _show_status(client: MCPClient, console: Console) -> None:
    """Show current day's status."""
    console.print("\n[dim]Fetching status...[/dim]")

    try:
        status = await get_today_status(client)
    except Exception as e:
        console.print(f"[red]Error fetching status:[/red] {e}")
        return

    # Build status display
    table = Table(title=f"📅 Today: {status['date']}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    # Events - show error if present
    calendar_error = status.get("calendar_error")
    if calendar_error:
        table.add_row("Calendar Events", f"[yellow]Error: {calendar_error}[/yellow]")
    else:
        events_count = status.get("events_count", 0)
        table.add_row("Calendar Events", str(events_count))

    # Time entry
    entry = status.get("time_entry")
    if entry:
        hours = entry.get("time_hours", 0)
        table.add_row("Logged Time", f"{hours:.1f}h")
        table.add_row("Entry ID", entry.get("id", "N/A"))
    else:
        table.add_row("Logged Time", "[yellow]No entry yet[/yellow]")

    console.print(table)

    # Show events
    events = status.get("events", [])
    if events:
        console.print("\n[bold]Events:[/bold]")
        for event in events:
            console.print(
                f"  • {event['summary']} [dim]({event['start']}-{event['end']})[/dim]"
            )

    # Show current note
    if entry and entry.get("note"):
        console.print("\n[bold]Current Note:[/bold]")
        # Strip HTML for display
        note = entry["note"]
        note = note.replace("<ul>", "").replace("</ul>", "")
        note = note.replace("<li>", "  • ").replace("</li>", "")
        note = note.replace("<em>", "").replace("</em>", "")
        console.print(f"[dim]{note.strip()}[/dim]")


async def _handle_sync(client: MCPClient, console: Console) -> None:
    """Handle sync command."""
    console.print("\n[dim]Syncing calendar to Productive...[/dim]")

    try:
        result = await sync_today(client)

        if result.success:
            console.print(
                f"[green]✓[/green] {result.action.capitalize()}: {result.message}"
            )
            if result.note_preview:
                console.print(f"\n[dim]Note preview:[/dim]")
                # Simple HTML display
                preview = result.note_preview[:200]
                if len(result.note_preview) > 200:
                    preview += "..."
                console.print(f"[dim]{preview}[/dim]")
        else:
            console.print(f"[red]✗[/red] {result.message}")

    except Exception as e:
        console.print(f"[red]Error during sync:[/red] {e}")


async def _handle_add(client: MCPClient, console: Console, item: str) -> None:
    """Handle add command."""
    console.print(f"\n[dim]Adding item: {item}[/dim]")

    try:
        result = await add_item_to_today(client, item)

        if result.success:
            console.print(f"[green]✓[/green] {result.message}")
        else:
            console.print(f"[red]✗[/red] {result.message}")

    except Exception as e:
        console.print(f"[red]Error adding item:[/red] {e}")


async def _handle_hours(client: MCPClient, console: Console) -> None:
    """Handle hours command."""
    console.print("\n[dim]Fetching your hours...[/dim]")

    try:
        data = await client.get_my_hours()

        # Use format_for_display for resilient parsing
        formatted = format_for_display(data, "hours")

        if formatted["success"]:
            console.print(f"\n[bold]Hours Summary:[/bold]")
            console.print(formatted["display_text"])

            # Show breakdown if available
            extracted = formatted.get("extracted", {})
            if extracted.get("total_hours"):
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Category")
                table.add_column("Hours", justify="right")

                table.add_row("Total", f"{extracted.get('total_hours', 0):.1f}h")
                if extracted.get("client_hours"):
                    table.add_row("Client", f"{extracted['client_hours']:.1f}h")
                if extracted.get("internal_hours"):
                    table.add_row("Internal", f"{extracted['internal_hours']:.1f}h")

                console.print(table)
        else:
            console.print(f"[yellow]Could not parse hours data[/yellow]")
            console.print(f"[dim]Raw: {str(data)[:200]}[/dim]")

    except Exception as e:
        console.print(f"[red]Error fetching hours:[/red] {e}")


def _show_help(console: Console) -> None:
    """Show help message."""
    help_text = """
## Commands

| Command | Description |
|---------|-------------|
| `sync` | Sync calendar events to Productive |
| `show` | Show today's status |
| `add <item>` | Add an item to today's log |
| `hours` | Show your hours summary |
| `help` | Show this help |
| `quit` | Exit the chat |

## Examples

```
add Code review for PR #123
add Debugging authentication issue
sync
show
hours
```
"""
    console.print(Markdown(help_text))
