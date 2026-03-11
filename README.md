# 🦙 TimeLlama

CLI for Productive.io time tracking with calendar sync, team hours review, and local LLM support.

TimeLlama helps you manage time tracking in Productive.io—sync calendar events to time entries, review team hours for invoicing, and get AI-powered work summaries. Uses a local LLM (Ollama) so all your data stays on your machine.

## Features

- **Calendar Sync**: Automatically create/update Productive time entries from your calendar
- **Team Hours Review**: Check employee hours for billing periods with AI-generated summaries
- **Invoice Approval**: Interactive workflow to review and approve team hours
- **Interactive Chat**: REPL mode for managing time entries with LLM suggestions
- **Cron-Ready**: One-shot sync command for automation
- **Local LLM**: Uses Ollama for formatting—all data stays on your machine

## Installation

### Prerequisites

1. **Ollama** (for LLM features):
   ```bash
   # macOS
   brew install ollama

   # Start Ollama server
   ollama serve

   # Pull the model
   ollama pull llama3.2:3b
   ```

2. **Productive.io API access** with your API token

3. **Calendar ICS file** (exported from Google Calendar, Outlook, etc.)

### Install TimeLlama

```bash
# Via pipx (recommended - isolated environment)
pipx install timellama

# Or via pip
pip install timellama

# Or install from source
git clone https://github.com/achrabaszcz/timellama.git
cd timellama
pip install -e .
```

## Configuration

Create a `.env` file or set environment variables:

```bash
# Required: Productive.io API credentials
PRODUCTIVE_API_TOKEN=your_api_token
PRODUCTIVE_ORG_ID=your_org_id
PRODUCTIVE_USER_ID=your_user_id

# Optional: Calendar ICS URL (for calendar sync)
ICS_CALENDAR_URL=https://outlook.office365.com/owa/calendar/.../calendar.ics

# Optional: Billing period cutoff day (default: 10)
PRODUCTIVE_BILLING_CUTOFF_DAY=10

# Optional: Ollama settings
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
```

### Getting Your Productive Credentials

1. **API Token**: Go to Productive → Settings → API tokens
2. **Organization ID**: Found in your Productive URL: `app.productive.io/org-ID/...`
3. **User ID**: Use the Productive API or check your profile settings

### Setting Up Your Calendar

TimeLlama fetches calendar data from an ICS URL. Most calendar providers offer a shareable ICS link:

**Outlook/Office 365**:
1. Go to Outlook Calendar → Settings → Shared calendars
2. Publish your calendar and copy the ICS link
3. Set `ICS_CALENDAR_URL` to the link

**Google Calendar**:
1. Go to Google Calendar → Settings → Settings for my calendars
2. Select your calendar → Integrate calendar
3. Copy the "Secret address in iCal format"
4. Set `ICS_CALENDAR_URL` to the link

## Usage

### Check Configuration

```bash
timellama doctor
```

### Show Today's Status

```bash
timellama status
```

### Sync Calendar to Productive

```bash
# Sync today's calendar events
timellama sync

# Dry run - see what would happen
timellama sync --dry-run

# Include LLM suggestions for additional items
timellama sync --suggestions
```

### Interactive Chat Mode

```bash
timellama chat
```

Commands in chat mode:
- `sync` - Sync calendar to Productive
- `show` - Show current status
- `add <item>` - Add item to today's log
- `help` - Show help
- `quit` - Exit

### Add Item to Today's Log

```bash
timellama add Code review for PR 123
timellama add Debugging authentication issue
```

### Employee Hours

```bash
# Get employee hours for billing period
timellama hours "John Doe"

# Current calendar month
timellama hours "John" --period current

# Previous month
timellama hours "Jane" --period previous

# Custom date range
timellama hours "John" --after 2026-01-01 --before 2026-01-31

# With AI summary
timellama hours "John" --summary
```

### Interactive Approval Workflow

```bash
timellama approve
```

## Cron Setup

For automated daily sync:

```bash
# Edit crontab
crontab -e

# Add line (sync at 6 PM every weekday)
0 18 * * 1-5 /path/to/timellama sync >> /var/log/timellama.log 2>&1
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        timellama CLI                        │
├─────────────────────────────────────────────────────────────┤
│  Commands:                                                  │
│  - timellama chat           (interactive mode)              │
│  - timellama sync           (automated sync for cron)       │
│  - timellama status         (show today's log)              │
│  - timellama hours <name>   (employee work summary)         │
│  - timellama approve        (review team hours)             │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Ollama    │      │ ics-calendar│      │ productive- │
│  (local)    │      │    MCP      │      │  time MCP   │
│ llama3.2:3b │      │ (bundled)   │      │  (bundled)  │
└─────────────┘      └─────────────┘      └─────────────┘
```

**Bundled MCP Servers**: The `productive-time-mcp` and `ics-calendar-mcp` servers are installed as dependencies. TimeLlama spawns them automatically—no separate setup needed.

## Privacy

- All data stays local
- Ollama runs on localhost:11434
- MCP servers run locally (stdio)
- Only Productive API calls go external (required for time tracking)
- No calendar data sent to cloud LLMs

## Development

```bash
# Clone repo
git clone https://github.com/achrabaszcz/timellama.git
cd timellama

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
```

## License

MIT
