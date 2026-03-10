"""Interactive chat mode for TimeLlama."""

import asyncio
from typing import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from timellama.mcp_client import MCPClient
from timellama.ollama_client import check_ollama_available, parse_chat_input
from timellama.sync import add_item_to_today, get_today_status, sync_today


async def chat_loop(client: MCPClient, console: Console) -> None:
    """Run the interactive chat loop.

    Args:
        client: Connected MCP client
        console: Rich console for output
    """
    # Check Ollama status
    ollama_ok, ollama_error = check_ollama_available()

    # Welcome message
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TimeLlama[/bold cyan] - Interactive Time Tracking\n\n"
            "Commands: [green]sync[/green], [green]show[/green], [green]add <item>[/green], "
            "[green]help[/green], [green]quit[/green]",
            title="🦙 Welcome",
        )
    )

    if not ollama_ok:
        console.print(
            f"[yellow]⚠ Ollama not available:[/yellow] {ollama_error}\n"
            "[dim]Some features will use basic formatting.[/dim]\n"
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

        # Parse input
        parsed = parse_chat_input(user_input)
        intent = parsed.get("intent", "unknown")
        data = parsed.get("data")

        # Handle commands
        if intent == "quit":
            console.print("[dim]Goodbye![/dim]")
            break

        elif intent == "help":
            _show_help(console)

        elif intent == "sync":
            await _handle_sync(client, console)

        elif intent == "show":
            await _show_status(client, console)

        elif intent == "add" and data:
            await _handle_add(client, console, data)

        elif intent == "remove" and data:
            console.print(
                "[yellow]Remove functionality not yet implemented.[/yellow]\n"
                "[dim]Use Productive UI to remove items.[/dim]"
            )

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

    # Events
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


def _show_help(console: Console) -> None:
    """Show help message."""
    help_text = """
## Commands

| Command | Description |
|---------|-------------|
| `sync` | Sync calendar events to Productive |
| `show` | Show today's status |
| `add <item>` | Add an item to today's log |
| `help` | Show this help |
| `quit` | Exit the chat |

## Examples

```
add Code review for PR #123
add Debugging authentication issue
sync
show
```
"""
    console.print(Markdown(help_text))
