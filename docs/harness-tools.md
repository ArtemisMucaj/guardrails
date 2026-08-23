# Harness tool reference

Tool names declared by each AI coding harness in the `tools` array of chat
completion requests. Used by the proxy to apply semantic preconditions (e.g.
intercepting write-only tools when the target file already exists).

## Claude Code

| Tool | Purpose |
|------|---------|
| `Read` | Read a file |
| `Write` | Create a file (write-only — file must not exist) |
| `Edit` | Edit an existing file |
| `MultiEdit` | Apply multiple edits to an existing file |
| `Bash` | Run a shell command |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents |
| `LS` | List directory contents |
| `WebFetch` | Fetch a URL |
| `WebSearch` | Search the web |
| `TodoRead` | Read the task list |
| `TodoWrite` | Write the task list |

## OpenCode

| Tool | Purpose |
|------|---------|
| `read` | Read a file |
| `write` | Create a file (write-only — file must not exist) |
| `edit` | Edit an existing file |
| `apply_patch` | Apply a unified diff patch |
| `list` | List directory contents |
| `grep` | Search file contents |
| `delete` | Delete a file |
| `bash` | Run a shell command |

## Pi (earendil-works/pi)

| Tool | Purpose |
|------|---------|
| `read` | Read a file |
| `write` | Create a file (write-only — file must not exist) |
| `edit` | Edit an existing file |
| `bash` | Run a shell command |

## Zed AI

| Tool | Purpose |
|------|---------|
| `read_file` | Read a file |
| `write_file` | Create a file (write-only — file must not exist) |
| `edit_file` | Edit an existing file |
| `create_directory` | Create a directory |
| `copy_path` | Copy a file or directory |
| `move_path` | Move a file or directory |
| `delete_path` | Delete a file or directory |

## GitHub Copilot CLI

| Tool | Purpose |
|------|---------|
| `read` | Read a file |
| `create` | Create a file (write-only — file must not exist) |
| `edit` | Edit an existing file |
| `shell` | Run a shell command |
| `grep` | Search file contents |
| `glob` | Find files by pattern |
| `web_fetch` | Fetch a URL |
| `web_search` | Search the web |

---

## Precondition-checked tools

Two rules run before the repair/validate loop. A call that fails either is
returned to the model as plain assistant text, never forwarded to the harness.

### Write-only tools — must not target an existing path

The proxy intercepts any call to the following tools when the target file
already exists, and instructs the model to read the file first and then use
the corresponding edit tool:

| Tool name | Harness |
|-----------|---------|
| `Write` | Claude Code |
| `write` | OpenCode, Pi |
| `write_file` | Zed AI |
| `create` | GitHub Copilot CLI |

### Edit tools — must follow a read of the same path

An in-place edit is only as good as the model's knowledge of the file: the old
string it matches on, or the hunk it patches against, has to describe what is
actually on disk. The proxy scans the transcript the client sent and refuses an
edit whose target never appears in a read:

| Tool name | Harness |
|-----------|---------|
| `Edit` | Claude Code |
| `MultiEdit` | Claude Code |
| `edit` | OpenCode, Pi, GitHub Copilot CLI |
| `apply_patch` | OpenCode |
| `edit_file` | Zed AI |

Only whole-file readers count as a read — `Read`, `read`, `read_file`. A `Grep`
returns matching lines and a `Glob` returns names, so neither tells the model
what an edit's context looks like.

The read set is rebuilt from each request's own `messages[]` (or Responses
`input[]`), so it is scoped to the conversation that sent it: a read in one chat
cannot license an edit in another.

### Failing open

The edit rule refuses only on positive evidence — a mutating call whose path is
absent from a transcript that is present and does contain recognisable tool
traffic. It stands down when the transcript is missing, carries no tool calls,
names its tools in an unknown vocabulary, or (on the Responses API) is a chained
turn whose history lives on the backend.

That is deliberate. A client may trim or summarise history, and the read may
well have happened where the scan cannot see it. A missed refusal costs one bad
edit the model can still be corrected on; a false refusal tells the model to
read a file it already read, and burns the turn.
