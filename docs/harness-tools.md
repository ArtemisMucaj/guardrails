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

## Write-only tools (precondition-checked)

The proxy intercepts any call to the following tools when the target file
already exists, and instructs the model to read the file first and then use
the corresponding edit tool:

| Tool name | Harness |
|-----------|---------|
| `Write` | Claude Code |
| `write` | OpenCode, Pi |
| `write_file` | Zed AI |
| `create` | GitHub Copilot CLI |
