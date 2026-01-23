# Claude Code Project Guide

## Issue Tracker: `br` (beads)

This project uses `br` (beads), an agent-first issue tracker with SQLite + JSONL backend.

### Workspace Info

- **Location**: `.beads/` directory
- **Database**: `.beads/beads.db`
- **Issue prefix**: `bd-` (e.g., `bd-123`)

### Essential Commands

#### Viewing Issues

```bash
# List ready issues (unblocked, not deferred) - START HERE
br ready

# List all open issues
br list

# Show specific issue details
br show <id>

# Search issues
br search <query>

# Project statistics
br stats
```

#### Working on Issues

```bash
# Claim an issue (assigns to you + sets status to in_progress)
br update <id> --claim

# Update issue status
br update <id> -s in_progress
br update <id> -s blocked

# Close an issue
br close <id>
br close <id> -r "reason for closing"

# Close and see newly unblocked issues
br close <id> --suggest-next
```

#### Creating Issues

```bash
# Quick capture (returns ID only)
br q "Issue title"
br q "Bug title" -t bug -p P1

# Full create with options
br create "Title" -t task -p P2 -d "Description"
br create "Title" --parent <parent-id>  # Creates as child issue
```

#### Dependencies

```bash
# Add dependency: <issue> depends on <depends-on>
br dep add <issue> <depends-on>

# Remove dependency
br dep remove <issue> <depends-on>

# View dependency tree
br dep tree <id>

# Check for cycles
br dep cycles
```

#### Labels

```bash
# Add/remove labels
br label add <id> label1,label2
br label remove <id> label1

# List all labels
br label list-all
```

### Issue Types

- `task` - Standard work item
- `bug` - Bug fix
- `feature` - New feature
- `epic` - Container for related issues

### Priorities

- `P0` / `0` - Critical
- `P1` / `1` - High
- `P2` / `2` - Medium (default)
- `P3` / `3` - Low
- `P4` / `4` - Backlog

### Workflow

When working on tickets, follow this process:

1. **Select a ticket**: Run `br ready` to see unblocked issues sorted by priority
2. **Review the ticket**: Run `br show <id>` to understand requirements and acceptance criteria
3. **Claim the ticket**: Run `br update <id> --claim` to mark it in progress
4. **Implement the changes**: Write code, tests, and documentation as needed
5. **Commit the changes**: Create a git commit with the ticket ID in the message (e.g., `(bd-2sh.2.1)`)
6. **Close the ticket**: Run `br close <id> -r "Brief summary of what was done"` or `br close <id> --suggest-next`

Example workflow:
```bash
br ready                           # Find next ticket
br show bd-2sh.2.1                 # Review requirements
br update bd-2sh.2.1 --claim       # Claim it
# ... implement changes ...
git add . && git commit -m "Add feature X (bd-2sh.2.1)"
br close bd-2sh.2.1 -r "Implemented feature X with tests"
```

### JSON Output

Add `--json` to any command for machine-readable output:

```bash
br ready --json
br show <id> --json
br list --json
```

### Deferring Issues

```bash
# Defer until a specific time
br defer <id> --until tomorrow
br defer <id> --until "+1h"
br defer <id> --until "2025-02-01"

# Undefer
br undefer <id>
```

### Epics

```bash
# Show epic status (progress of children)
br epic status

# Auto-close eligible epics (all children closed)
br epic close-eligible
```

### Comments

```bash
# Add comment to issue
br comments add <id> "Comment text"

# List comments
br comments list <id>
```

### Sync (JSONL)

The database auto-syncs with `.beads/issues.jsonl`. Manual sync:

```bash
br sync --status        # Check sync status
br sync --flush-only    # Export DB to JSONL
br sync --import-only   # Import JSONL to DB
```

### Useful Filters

```bash
# Filter by type
br list -t bug
br list -t epic

# Filter by priority
br list -p P0 -p P1

# Filter by label
br list -l "backend"

# Filter by assignee
br list --assignee "name"
br list --unassigned

# Include closed issues
br list -a

# Blocked issues
br blocked
```
