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
2. **Gather context before starting**:
   - Run `br show <id>` to understand requirements and acceptance criteria
   - Read the **parent ticket** (`br show <parent-id>`) to understand the broader goal
   - Read any **referenced plan documents** (e.g., `docs/RWKV7_HIP_BACKEND_PLAN.md`) and follow the architectural decisions specified there
   - Read **comments on sibling/predecessor tickets** (`br comments list <sibling-id>`) — closed tickets often contain implementation notes, divergences from the plan, or context that affects your work
   - If anything is unclear, ask the user for clarification before starting
3. **Claim the ticket**: Run `br update <id> --claim` to mark it in progress
4. **Implement the changes**: Write code, tests, and documentation as needed
5. **Comment on divergences**: If the implementation diverges from the plan or ticket description (e.g., different approach needed, unexpected dependency, extra work required), add a comment explaining why: `br comments add <id> "Diverged from plan: <reason>"`
6. **Verify acceptance criteria**: Ensure ALL acceptance criteria in the ticket are met
7. **Commit the changes**: Create a git commit with the ticket ID in the message (e.g., `(bd-2sh.2.1)`)
8. **Commit ticket updates**: After any `br` commands that modify ticket state (close, comment, update), commit `.beads/issues.jsonl` to keep the tracker in sync:
   ```bash
   git add .beads/issues.jsonl && git commit -m "Update beads"
   ```
9. **Close with a summary comment**: Add a comment summarizing what was done, then close:
   ```bash
   br comments add <id> "Summary of changes and any notes for downstream tickets"
   br close <id> --suggest-next
   ```

**CRITICAL - Before Closing a Ticket:**
- [ ] All code changes are **committed** (check `git status` - no uncommitted work)
- [ ] All acceptance criteria are **verified and met**
- [ ] Tests pass at the tolerances specified in the ticket/plan
- [ ] Implementation follows architectural decisions from referenced plan documents
- [ ] A closing comment has been added summarizing the work and any divergences

**Do NOT close a ticket if:**
- There is uncommitted work in the working directory
- Acceptance criteria checkboxes are not satisfied
- Tests pass only with loosened tolerances (document the gap instead)

Example workflow:
```bash
br ready                           # Find next ticket
br show bd-2sh.2.1                 # Review requirements
br show bd-2sh.2                   # Read parent for broader context
br comments list bd-2sh.1.3        # Check notes from predecessor ticket
# Read any referenced plan docs!
br update bd-2sh.2.1 --claim       # Claim it
# ... implement changes ...
# If diverging from plan:
br comments add bd-2sh.2.1 "Had to also update X because Y"
git status                         # Verify no uncommitted changes remain
git add . && git commit -m "Add feature X (bd-2sh.2.1)"
br comments add bd-2sh.2.1 "Done: implemented X with tests. Note: Z for downstream."
br close bd-2sh.2.1 --suggest-next
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

## Python Environment

Use `uv` for all Python package management (faster than pip):

```bash
# Install packages
uv pip install <package>

# Install from requirements
uv pip install -r requirements.txt

# Run Python scripts
python3 scripts/foo.py
```

## Backward Compatibility

Do NOT maintain backward compatibility unless explicitly requested. Delete old APIs, remove deprecated wrappers, and clean up dead code. Don't keep old code paths "just in case."

## Parallel Ticket Work with Git Worktrees

When an orchestrator agent farms out multiple tickets to sub-agents in parallel, each agent **MUST** work in its own git worktree to avoid stepping on each other's changes. This prevents:
- One agent committing another agent's uncommitted changes
- Test runs picking up partial changes from a sibling agent
- Merge conflicts from concurrent edits to the same file

### Orchestrator responsibilities

The orchestrator (parent agent) manages the worktree lifecycle:

```bash
# Before launching parallel agents, create worktrees from the current branch:
git worktree add /tmp/worktree-bd-TICKET_A hip   # or whatever the current branch is
git worktree add /tmp/worktree-bd-TICKET_B hip

# After agents complete, rebase their commits onto the main branch:
# (orchestrator reviews, resolves conflicts, and integrates)
```

### Agent instructions (include in agent prompts)

Tell each agent:

1. **Work in your assigned worktree** — all file reads, writes, and git operations happen in the worktree directory (e.g., `/tmp/worktree-bd-TICKET_A/`), NOT in `/workspace/web-rwkv/`.
2. **Create a feature branch** from the worktree:
   ```bash
   cd /tmp/worktree-bd-TICKET_A
   git checkout -b ticket/bd-TICKET_A
   ```
3. **Commit only your own changes** — since the worktree starts clean, `git add .` is safe.
4. **Run `cargo check` / tests from the worktree directory** so you only see your own changes.
5. **Do NOT touch `/workspace/web-rwkv/`** — that is the orchestrator's workspace.

### After agents complete

The orchestrator rebases branches back (keep linear history, no merge commits):

```bash
# From the main workspace:
cd /workspace/web-rwkv
git cherry-pick <commit-hash>   # cherry-pick each agent's commit(s) onto hip

# Clean up worktrees:
git worktree remove /tmp/worktree-bd-TICKET_A
git worktree remove /tmp/worktree-bd-TICKET_B
```

**IMPORTANT: Use rebase/cherry-pick, NOT merge.** We maintain linear history — no merge commits.

If there are conflicts, the orchestrator resolves them before proceeding to dependent tickets.

### When worktrees are NOT needed

- Sequential ticket work (one at a time) — just work in `/workspace/web-rwkv/` directly
- Tickets that touch completely disjoint files with no shared dependencies — still preferred to use worktrees for safety

## Code Refactoring: Extract, Don't Rewrite

When refactoring — moving code between files, extracting helper functions, or splitting modules — **never rewrite large blocks of code from scratch**. Instead:

1. **Use `sed` to extract exact line ranges** from the source file:
   ```bash
   # Extract lines 100-200 from step.rs to see the code block
   sed -n '100,200p' src/hip/model/step.rs
   ```

2. **Copy the extracted code** into the new file using the Write/Edit tools, preserving the original logic character-for-character.

3. **Add function signatures and parameter lists** around the extracted blocks — this is the only new code you write.

4. **Replace the original code** in the source file with calls to the new helpers using the Edit tool.

Why this matters:
- Rewriting 800 lines from memory introduces subtle transcription errors
- Extracting preserves exact logic, operator precedence, and edge cases
- It's faster, cheaper, and more reliable
- Reviewers can verify the refactor is behavior-preserving

**Anti-pattern** (do NOT do this):
```
Read step.rs, understand the logic, then write dispatch_helpers.rs from scratch
with your understanding of what the code does.
```

**Correct pattern**:
```
sed -n '150,220p' src/hip/model/step.rs > extracted block
Wrap extracted block in fn attention_block(...) { ... }
Edit step.rs to replace lines 150-220 with: attention_block(args)
```

## Platform Assumptions

Never assume what tools, hardware, SDKs, or runtime features are available. Always investigate the actual system capabilities before planning or writing code. Check installed toolchains, device properties, available APIs, and supported features rather than guessing based on prior knowledge.

## Compact Instructions

When context is compacted, preserve:
- The `br` issue tracker workflow (ready, show, claim, close)
- **Ticket closure requirements** (commit before close, verify acceptance criteria)
- **Parallel work requires git worktrees** — one worktree per agent, cherry-pick/rebase back (no merge commits, linear history)
- **Refactoring = extract, don't rewrite** — use sed to extract code, not retype from memory
- Current branch context and recent commits
- References to plan documents (e.g., `docs/RWKV7_HIP_BACKEND_PLAN.md`)
- Architectural decisions (e.g., column-major GEMM, tolerance specifications)
