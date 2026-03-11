"""Employee hours and invoice approval functionality."""

import asyncio
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from timellama.mcp_client import MCPClient
from timellama.ollama_client import (
    check_ollama_available,
    extract_action_data,
    format_for_display,
    summarize_work,
)


@dataclass
class HoursSummary:
    """Summary of an employee's hours."""

    name: str
    period_start: date
    period_end: date
    total_hours: float
    client_hours: float
    internal_hours: float
    holiday_hours: float
    entries: list[dict]
    summary: str | None = None


async def get_employee_hours(
    client: MCPClient,
    name: str,
    period: str | None = None,
    after: date | None = None,
    before: date | None = None,
) -> HoursSummary:
    """Get an employee's hours summary.

    Uses Ollama formatters for resilient parsing of various response formats.

    Args:
        client: Connected MCP client
        name: Employee name to search for
        period: Optional period ("current", "previous", or None for default billing)
        after: Optional start date
        before: Optional end date

    Returns:
        HoursSummary with hours breakdown
    """
    # Determine date range
    if after and before:
        start_date = after
        end_date = before
    elif period == "current":
        today = date.today()
        start_date = today.replace(day=1)
        # End of current month
        if today.month == 12:
            end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)
    elif period == "previous":
        today = date.today()
        # First day of previous month
        if today.month == 1:
            start_date = date(today.year - 1, 12, 1)
        else:
            start_date = date(today.year, today.month - 1, 1)
        # Last day of previous month
        end_date = today.replace(day=1) - timedelta(days=1)
    else:
        # Default: use MCP server's billing period
        start_date = None
        end_date = None

    # Fetch employee hours from MCP
    try:
        hours_data = await client.get_employee_hours(
            name=name,
            period=period,
            after=start_date,
            before=end_date,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch hours for {name}: {e}")

    # Use Ollama formatter for resilient parsing
    formatted = format_for_display(hours_data, "hours", context=f"Employee: {name}")

    if formatted.get("success") and formatted.get("extracted"):
        extracted = formatted["extracted"]

        # Get values from formatted extraction (handles various field names)
        total_hours = extracted.get("total_hours", 0)
        client_hours = extracted.get("client_hours", 0)
        internal_hours = extracted.get("internal_hours", 0)
        holiday_hours = extracted.get("holiday_hours", 0)
        entries = extracted.get("entries", [])

        # Also try to get from raw data for backward compatibility
        if isinstance(hours_data, dict):
            total_hours = total_hours or hours_data.get("total_hours", 0)
            client_hours = client_hours or hours_data.get("client_hours", 0)
            internal_hours = internal_hours or hours_data.get("internal_hours", 0)
            holiday_hours = holiday_hours or hours_data.get("holiday_hours", 0)
            entries = entries or hours_data.get("entries", [])

            # Handle 'time' field (minutes) if 'hours' not present
            if not total_hours and hours_data.get("time"):
                hours_result = extract_action_data(hours_data, "get_hours")
                if hours_result.get("success") and hours_result.get("data"):
                    total_hours = hours_result["data"]

            # Get period info
            period_info = hours_data.get("period", {})
            if period_info:
                try:
                    start_date = date.fromisoformat(period_info.get("start", str(date.today())))
                    end_date = date.fromisoformat(period_info.get("end", str(date.today())))
                except (ValueError, TypeError):
                    pass

        if not start_date:
            start_date = date.today().replace(day=1)
        if not end_date:
            end_date = date.today()

        return HoursSummary(
            name=hours_data.get("name", name) if isinstance(hours_data, dict) else name,
            period_start=start_date,
            period_end=end_date,
            total_hours=float(total_hours) if total_hours else 0,
            client_hours=float(client_hours) if client_hours else 0,
            internal_hours=float(internal_hours) if internal_hours else 0,
            holiday_hours=float(holiday_hours) if holiday_hours else 0,
            entries=entries if isinstance(entries, list) else [],
        )

    # Fallback: parse response directly (backward compatibility)
    if isinstance(hours_data, dict):
        total_hours = hours_data.get("total_hours", 0)
        client_hours = hours_data.get("client_hours", 0)
        internal_hours = hours_data.get("internal_hours", 0)
        holiday_hours = hours_data.get("holiday_hours", 0)
        entries = hours_data.get("entries", [])
        period_info = hours_data.get("period", {})

        if period_info:
            try:
                start_date = date.fromisoformat(period_info.get("start", str(date.today())))
                end_date = date.fromisoformat(period_info.get("end", str(date.today())))
            except (ValueError, TypeError):
                pass

        if not start_date:
            start_date = date.today().replace(day=1)
        if not end_date:
            end_date = date.today()

        return HoursSummary(
            name=hours_data.get("name", name),
            period_start=start_date,
            period_end=end_date,
            total_hours=float(total_hours) if total_hours else 0,
            client_hours=float(client_hours) if client_hours else 0,
            internal_hours=float(internal_hours) if internal_hours else 0,
            holiday_hours=float(holiday_hours) if holiday_hours else 0,
            entries=entries if isinstance(entries, list) else [],
        )

    # Final fallback for raw response
    return HoursSummary(
        name=name,
        period_start=start_date or date.today(),
        period_end=end_date or date.today(),
        total_hours=0,
        client_hours=0,
        internal_hours=0,
        holiday_hours=0,
        entries=[],
    )


def display_hours_summary(summary: HoursSummary, console: Console) -> None:
    """Display an employee's hours summary.

    Args:
        summary: Hours summary to display
        console: Rich console for output
    """
    # Header
    console.print()
    console.print(
        Panel.fit(
            f"[bold]{summary.name}[/bold]\n"
            f"[dim]{summary.period_start} to {summary.period_end}[/dim]",
            title="📊 Hours Summary",
        )
    )

    # Hours table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Category")
    table.add_column("Hours", justify="right")
    table.add_column("Percentage", justify="right")

    total = summary.total_hours or 1  # Avoid division by zero

    table.add_row(
        "Client Work",
        f"{summary.client_hours:.1f}h",
        f"{(summary.client_hours / total * 100):.0f}%",
    )
    table.add_row(
        "Internal",
        f"{summary.internal_hours:.1f}h",
        f"{(summary.internal_hours / total * 100):.0f}%",
    )
    table.add_row(
        "Holiday/PTO",
        f"{summary.holiday_hours:.1f}h",
        f"{(summary.holiday_hours / total * 100):.0f}%",
    )
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{summary.total_hours:.1f}h[/bold]",
        "[bold]100%[/bold]",
    )

    console.print(table)

    # Work breakdown by notes/projects
    if summary.entries:
        console.print("\n[bold]Work Breakdown:[/bold]")

        # Group entries by project/service
        by_project: dict[str, float] = {}
        for entry in summary.entries:
            project = entry.get("project", entry.get("service", "Other"))
            hours = entry.get("hours", 0)
            by_project[project] = by_project.get(project, 0) + hours

        # Sort by hours descending
        sorted_projects = sorted(by_project.items(), key=lambda x: x[1], reverse=True)

        for project, hours in sorted_projects[:10]:
            bar_width = int((hours / total) * 30)
            bar = "█" * bar_width + "░" * (30 - bar_width)
            console.print(f"  {project[:30]:<30} [{bar}] {hours:.1f}h")


async def generate_work_summary(
    summary: HoursSummary, console: Console
) -> str | None:
    """Generate an LLM summary of work for invoice approval.

    Args:
        summary: Hours summary with entries
        console: Rich console for output

    Returns:
        Generated summary text or None
    """
    available, error = check_ollama_available()
    if not available:
        console.print(f"[yellow]⚠ Ollama not available:[/yellow] {error}")
        console.print("[dim]Cannot generate AI summary.[/dim]")
        return None

    console.print("\n[dim]Generating AI summary...[/dim]")

    # Prepare entries text
    entries_text = []
    for entry in summary.entries:
        note = entry.get("note", "")
        project = entry.get("project", "")
        hours = entry.get("hours", 0)
        entries_text.append(f"- {project}: {note} ({hours}h)")

    entries_str = "\n".join(entries_text[:50])  # Limit to 50 entries

    hours_data = {
        "total_hours": summary.total_hours,
        "client_hours": summary.client_hours,
        "internal_hours": summary.internal_hours,
    }

    summary_text = summarize_work(entries_str, hours_data)

    console.print("\n[bold]AI Summary:[/bold]")
    console.print(Panel(summary_text, border_style="green"))

    return summary_text


async def interactive_approval(client: MCPClient, console: Console) -> None:
    """Interactive workflow for reviewing team hours.

    Args:
        client: Connected MCP client
        console: Rich console for output
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Team Hours Review[/bold cyan]\n\n"
            "Review employee hours for invoice approval.",
            title="🦙 TimeLlama Approve",
        )
    )

    while True:
        # Get employee name
        try:
            name = Prompt.ask(
                "\n[bold]Employee name[/bold] (or 'quit' to exit)",
                default="",
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting approval workflow.[/dim]")
            break

        if not name or name.lower() in ("quit", "exit", "q"):
            console.print("[dim]Exiting approval workflow.[/dim]")
            break

        # Get period
        period = Prompt.ask(
            "[bold]Period[/bold]",
            choices=["billing", "current", "previous"],
            default="billing",
        )

        period_arg = None if period == "billing" else period

        # Fetch hours
        console.print(f"\n[dim]Fetching hours for {name}...[/dim]")

        try:
            summary = await get_employee_hours(client, name, period=period_arg)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            continue

        # Display summary
        display_hours_summary(summary, console)

        # Generate AI summary
        if summary.entries:
            generate_summary = Confirm.ask(
                "\nGenerate AI summary?",
                default=True,
            )
            if generate_summary:
                await generate_work_summary(summary, console)

        # Ask if approved
        console.print()
        approved = Confirm.ask(
            f"[bold]Approve {summary.total_hours:.1f}h for {summary.name}?[/bold]",
            default=True,
        )

        if approved:
            console.print(f"[green]✓[/green] Approved {summary.name}'s hours.")
        else:
            console.print(f"[yellow]⚠[/yellow] Not approved. Review needed.")

        # Continue with another employee?
        continue_review = Confirm.ask("\nReview another employee?", default=True)
        if not continue_review:
            break

    console.print("\n[dim]Approval workflow complete.[/dim]")
