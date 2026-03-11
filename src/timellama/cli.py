"""CLI commands for TimeLlama using Click."""

import asyncio
import os
import sys
from datetime import date, datetime

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from timellama.mcp_client import MCPClient
from timellama.ollama_client import check_model_available, check_ollama_available


# Load environment variables
load_dotenv()

# Rich console for output
console = Console()


def validate_environment(require_calendar: bool = False) -> tuple[bool, list[str]]:
    """Validate required environment variables.

    Args:
        require_calendar: If True, ICS_CALENDAR_URL is required

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    required_vars = [
        ("PRODUCTIVE_API_TOKEN", "Productive API token"),
        ("PRODUCTIVE_ORG_ID", "Productive organization ID"),
        ("PRODUCTIVE_USER_ID", "Productive user ID"),
    ]

    for var, description in required_vars:
        if not os.environ.get(var):
            errors.append(f"Missing {var} ({description})")

    # Check calendar URL if required
    if require_calendar:
        calendar_url = os.environ.get("ICS_CALENDAR_URL", "")
        if not calendar_url:
            errors.append("Missing ICS_CALENDAR_URL (Calendar URL)")

    return len(errors) == 0, errors


def check_prerequisites(require_ollama: bool = False, require_calendar: bool = False) -> bool:
    """Check all prerequisites are met.

    Args:
        require_ollama: If True, fail if Ollama is not available
        require_calendar: If True, fail if calendar URL is not configured

    Returns:
        True if prerequisites are met
    """
    # Check environment
    env_ok, errors = validate_environment(require_calendar=require_calendar)
    if not env_ok:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  • {error}")
        console.print("\n[dim]See .env.example for required variables.[/dim]")
        return False

    # Check Ollama
    ollama_ok, ollama_error = check_ollama_available()
    if not ollama_ok:
        if require_ollama:
            console.print(f"[red]Ollama not available:[/red] {ollama_error}")
            console.print("[dim]Start Ollama with: ollama serve[/dim]")
            return False
        else:
            console.print(f"[yellow]⚠ Ollama not available:[/yellow] {ollama_error}")
            console.print("[dim]Some features will use basic formatting.[/dim]\n")
    else:
        # Check model
        model_ok, model_error = check_model_available()
        if not model_ok:
            console.print(f"[yellow]⚠ Model not available:[/yellow] {model_error}")
            model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
            console.print(f"[dim]Pull model with: ollama pull {model}[/dim]\n")

    return True


@click.group()
@click.version_option(version="0.2.2", prog_name="timellama")
def main():
    """🦙 TimeLlama - Productive.io time tracking CLI.

    Manage time entries, sync calendar events, review team hours for invoicing,
    and get AI-powered work summaries—all with local LLM support.
    """
    pass


@main.command()
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes")
@click.option("--suggestions", is_flag=True, help="Include LLM-generated suggestions")
def sync(dry_run: bool, suggestions: bool):
    """Sync today's calendar events to Productive.

    Fetches calendar events and creates/updates a time entry in Productive.
    Ideal for running via cron for automated time logging.
    """
    if not check_prerequisites(require_calendar=True):
        sys.exit(1)

    async def _sync():
        from timellama.sync import sync_today

        client = MCPClient()
        async with client.connect():
            result = await sync_today(
                client, dry_run=dry_run, include_suggestions=suggestions
            )

            if result.success:
                icon = "✓" if not dry_run else "ℹ"
                console.print(f"[green]{icon}[/green] {result.message}")
                if result.note_preview:
                    console.print(f"\n[dim]Note:[/dim]")
                    # Simple HTML display
                    preview = result.note_preview.replace("<ul>", "").replace("</ul>", "")
                    preview = preview.replace("<li>", "  • ").replace("</li>", "\n")
                    preview = preview.replace("<em>", "").replace("</em>", "")
                    console.print(f"[dim]{preview.strip()}[/dim]")
            else:
                console.print(f"[red]✗[/red] {result.message}")
                sys.exit(1)

    asyncio.run(_sync())


@main.command()
def status():
    """Show today's time entry status.

    Displays current calendar events and logged time for today.
    """
    if not check_prerequisites():
        sys.exit(1)

    async def _status():
        from timellama.sync import get_today_status
        from rich.table import Table

        client = MCPClient()
        async with client.connect():
            data = await get_today_status(client)

            # Build status display
            table = Table(title=f"📅 Today: {data['date']}", show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value")

            # Events - show error if present
            calendar_error = data.get("calendar_error")
            if calendar_error:
                table.add_row("Calendar Events", f"[yellow]Error: {calendar_error}[/yellow]")
            else:
                table.add_row("Calendar Events", str(data.get("events_count", 0)))

            # Time entry
            entry = data.get("time_entry")
            if entry:
                hours = entry.get("time_hours", 0)
                table.add_row("Logged Time", f"{hours:.1f}h")
                table.add_row("Entry ID", entry.get("id", "N/A"))
            else:
                table.add_row("Logged Time", "[yellow]No entry yet[/yellow]")

            console.print(table)

            # Show events
            events = data.get("events", [])
            if events:
                console.print("\n[bold]Calendar Events:[/bold]")
                for event in events:
                    console.print(
                        f"  • {event['summary']} [dim]({event['start']}-{event['end']})[/dim]"
                    )

            # Show logged note content
            if entry and entry.get("note"):
                console.print("\n[bold]Logged in Productive:[/bold]")
                note = entry["note"]
                # Strip HTML for display
                note = note.replace("<ul>", "").replace("</ul>", "")
                note = note.replace("<li>", "  • ").replace("</li>", "")
                note = note.replace("<p>", "").replace("</p>", "")
                note = note.replace("<em>", "").replace("</em>", "")
                note = note.replace("\n\n", "\n").strip()
                console.print(f"[green]{note}[/green]")

    asyncio.run(_status())


@main.command()
def chat():
    """Start interactive chat mode.

    Interactive REPL for managing time entries with LLM-powered suggestions.
    """
    if not check_prerequisites():
        sys.exit(1)

    async def _chat():
        from timellama.chat import chat_loop

        client = MCPClient()
        async with client.connect():
            await chat_loop(client, console)

    asyncio.run(_chat())


@main.command()
@click.argument("name")
@click.option(
    "--period",
    type=click.Choice(["billing", "current", "previous"]),
    default="billing",
    help="Time period for hours summary",
)
@click.option("--after", type=click.DateTime(formats=["%Y-%m-%d"]), help="Start date (YYYY-MM-DD)")
@click.option("--before", type=click.DateTime(formats=["%Y-%m-%d"]), help="End date (YYYY-MM-DD)")
@click.option("--summary", is_flag=True, help="Generate AI summary of work")
def hours(name: str, period: str, after: datetime | None, before: datetime | None, summary: bool):
    """Get employee's hours summary for invoice approval.

    NAME is the employee name to search for.

    Examples:

        timellama hours "John Doe"

        timellama hours "John" --period current

        timellama hours "Jane" --after 2026-01-01 --before 2026-01-31
    """
    if not check_prerequisites():
        sys.exit(1)

    async def _hours():
        from timellama.hours import display_hours_summary, generate_work_summary, get_employee_hours

        period_arg = None if period == "billing" else period
        after_date = after.date() if after else None
        before_date = before.date() if before else None

        client = MCPClient()
        async with client.connect():
            try:
                hours_summary = await get_employee_hours(
                    client,
                    name=name,
                    period=period_arg,
                    after=after_date,
                    before=before_date,
                )
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                sys.exit(1)

            display_hours_summary(hours_summary, console)

            if summary and hours_summary.entries:
                await generate_work_summary(hours_summary, console)

    asyncio.run(_hours())


@main.command()
def approve():
    """Interactive review of team hours for invoices.

    Walks through employee hours with AI-generated summaries
    for invoice approval workflow.
    """
    if not check_prerequisites():
        sys.exit(1)

    async def _approve():
        from timellama.hours import interactive_approval

        client = MCPClient()
        async with client.connect():
            await interactive_approval(client, console)

    asyncio.run(_approve())


@main.command()
@click.argument("item", nargs=-1, required=True)
def add(item: tuple[str, ...]):
    """Add an item to today's time log.

    ITEM is the text to add to the time entry note.

    Example:

        timellama add Code review for PR 123
    """
    if not check_prerequisites():
        sys.exit(1)

    item_text = " ".join(item)

    async def _add():
        from timellama.sync import add_item_to_today

        client = MCPClient()
        async with client.connect():
            result = await add_item_to_today(client, item_text)

            if result.success:
                console.print(f"[green]✓[/green] {result.message}")
            else:
                console.print(f"[red]✗[/red] {result.message}")
                sys.exit(1)

    asyncio.run(_add())


@main.command()
def doctor():
    """Check system configuration and dependencies.

    Validates environment variables, Ollama availability, and MCP server connections.
    """
    console.print()
    console.print(Panel.fit("[bold]TimeLlama Doctor[/bold]", title="🩺"))
    console.print()

    all_ok = True

    # Check environment variables
    console.print("[bold]Environment Variables:[/bold]")
    env_vars = [
        ("PRODUCTIVE_API_TOKEN", True),
        ("PRODUCTIVE_ORG_ID", True),
        ("PRODUCTIVE_USER_ID", True),
        ("PRODUCTIVE_SERVICE_ID", False),  # Optional but needed for creating entries
        ("ICS_CALENDAR_URL", False),  # Optional - only needed for calendar sync
        ("PRODUCTIVE_BILLING_CUTOFF_DAY", False),
        ("OLLAMA_MODEL", False),
        ("OLLAMA_HOST", False),
    ]

    for var, required in env_vars:
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if "TOKEN" in var or "SECRET" in var:
                display = value[:4] + "****"
            elif "URL" in var and len(value) > 50:
                display = value[:50] + "..."
            else:
                display = value
            console.print(f"  [green]✓[/green] {var}: {display}")
        elif required:
            console.print(f"  [red]✗[/red] {var}: [red]missing (required)[/red]")
            all_ok = False
        else:
            console.print(f"  [dim]○[/dim] {var}: [dim]not set (optional)[/dim]")

    # Check calendar URL
    console.print("\n[bold]Calendar:[/bold]")
    calendar_url = os.environ.get("ICS_CALENDAR_URL", "")
    if calendar_url:
        console.print(f"  [green]✓[/green] URL configured")
    else:
        console.print(f"  [dim]○[/dim] ICS_CALENDAR_URL not set (optional, needed for calendar sync)")

    # Check Ollama
    console.print("\n[bold]Ollama:[/bold]")
    ollama_ok, ollama_error = check_ollama_available()
    if ollama_ok:
        console.print(f"  [green]✓[/green] Server running")

        model_ok, model_error = check_model_available()
        model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        if model_ok:
            console.print(f"  [green]✓[/green] Model: {model}")
        else:
            console.print(f"  [yellow]⚠[/yellow] Model {model}: {model_error}")
    else:
        console.print(f"  [yellow]⚠[/yellow] Not running: {ollama_error}")
        console.print(f"  [dim]   Start with: ollama serve[/dim]")

    # Check MCP servers
    console.print("\n[bold]MCP Servers:[/bold]")

    async def _check_mcp():
        nonlocal all_ok
        from timellama.mcp_client import test_connection

        prod_ok, cal_ok, error = await test_connection()

        if prod_ok:
            console.print(f"  [green]✓[/green] productive-time-mcp")
        else:
            console.print(f"  [red]✗[/red] productive-time-mcp: connection failed")
            all_ok = False

        if cal_ok:
            console.print(f"  [green]✓[/green] ics-calendar-mcp")
        else:
            console.print(f"  [red]✗[/red] ics-calendar-mcp: connection failed")
            all_ok = False

        if error:
            console.print(f"  [dim]{error}[/dim]")

    try:
        asyncio.run(_check_mcp())
    except Exception as e:
        console.print(f"  [red]✗[/red] MCP connection error: {e}")
        all_ok = False

    # Summary
    console.print()
    if all_ok:
        console.print("[green]✓ All checks passed![/green]")
    else:
        console.print("[red]✗ Some checks failed. See above for details.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
