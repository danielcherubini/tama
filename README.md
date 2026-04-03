<div align="center">

<img src="https://raw.githubusercontent.com/danielcherubini/kronk/main/icon.png" alt="Kronk" height="96" />

# KRONK

> Oh yeah, it's all coming together.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Crates.io](https://img.shields.io/crates/v/kronk.svg)](https://crates.io/crates/kronk)
[![Build Status](https://img.shields.io/github/actions/workflow/status/danielcherubini/kronk/ci.yml?label=CI&style=flat-square)](https://github.com/danielcherubini/kronk/actions)

</div>

A local AI server written in Rust. Kronk provides an OpenAI-compatible API on a single port, automatically managing backend lifecycles — starting models on demand, routing requests, and unloading idle models to save resources.

Think of it as your own local Ollama or LM Studio server, but for llama.cpp and ik_llama backends.

> [!TIP]
> Get up and running: `kronk model pull bartowski/OmniCoder-8B-GGUF && kronk serve`

---

## Quick Start

### Install

**Windows:** Download the installer from [Releases](https://github.com/danielcherubini/kronk/releases), or:

```bash
cargo install --git https://github.com/danielcherubini/kronk kronk
```

**Linux (Debian/Ubuntu):**

```bash
sudo dpkg -i kronk_*.deb
```

**Linux (Fedora/RHEL):**

```bash
sudo rpm -i kronk-*.rpm
```

### Pull a model from HuggingFace

```bash
kronk model pull bartowski/OmniCoder-8B-GGUF
```

Kronk downloads all available quants, detects your GPU VRAM, and suggests optimal context sizes.

### Start the server

```bash
kronk serve
```

That's it. Kronk starts an OpenAI-compatible server on `http://localhost:11434`. When a request comes in for a model, Kronk automatically starts the right backend, waits for it to be ready, and forwards the request.

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bartowski/OmniCoder-8B-GGUF",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Models are unloaded after 5 minutes of inactivity (configurable with `--idle-timeout`).

### Install as a system service

```bash
# Install and start (run as admin / sudo)
kronk service install
kronk service start

# After that, no admin needed
kronk service stop
kronk service start
kronk status
```

> [!NOTE]
> For debugging individual backends, you can still use `kronk run <server-name>` to run a single server in the foreground.

---

## Web Control Plane

Kronk includes a web-based control plane UI for managing models, viewing logs, and editing config from a browser.

```bash
# 1. Build the frontend (requires trunk: cargo install trunk)
cd crates/kronk-web && trunk build --release && cd ../..

# 2. Start the web server (port 11435 by default)
cargo run --package kronk-web --features ssr

# Or via the CLI (with web-ui feature):
cargo run --package kronk --features web-ui -- web --port 11435

# 3. Open http://localhost:11435
```

The web UI proxies all `/kronk/v1/` requests to the running Kronk proxy (default `http://127.0.0.1:11434`). Configure with env vars:
- `KRONK_PROXY_URL` — proxy base URL (default: `http://127.0.0.1:11434`)
- `KRONK_LOGS_DIR` — path to Kronk log files (optional)
- `KRONK_CONFIG_PATH` — path to `kronk.toml` for config editor (optional)

The web server starts automatically alongside the proxy when using `kronk service start`.

---

## CLI

```text
kronk serve [--host H] [--port P] [--idle-timeout S]  Start the server
kronk status                                       Show status of all servers
kronk service install                              Install as a system service
kronk service start                                Start the service
kronk service stop                                 Stop the service
kronk service remove                               Remove the service
kronk model pull <repo>                            Pull a model from HuggingFace
kronk model ls                                     List installed models
kronk model ps                                     Show running model processes
kronk model create <name>                          Create a server from an installed model
kronk model rm <model>                             Remove an installed model
kronk model scan                                   Scan for untracked GGUF files
kronk model search <query>                         Search HuggingFace for GGUF models
kronk config show                                  Print the current configuration
kronk config edit                                  Open config file in editor
kronk config path                                  Show the config file path
kronk logs [name]                                  View logs (defaults to proxy logs)
kronk run <name> [--ctx N]                         Run a single backend (for debugging)
```

### Backend Management

Kronk manages LLM backend installations (llama.cpp, ik_llama) with automatic version tracking and updates:

```bash
kronk backend install llama_cpp    # Install latest llama.cpp
kronk backend install ik_llama     # Install latest ik_llama (builds from source)
kronk backend install llama_cpp --version b8407  # Install specific version
kronk backend install llama_cpp --build    # Force build from source
kronk backend update <name>        # Update to latest version
kronk backend list                 # List installed backends
kronk backend remove <name>        # Remove a backend
kronk backend check-updates        # Check for updates
```

### Installation Details

- **llama.cpp**: Downloads pre-built binaries for your platform, or builds from source with GPU support
- **ik_llama**: Always builds from source (no pre-built binaries available)
- **Linux/macOS:** Backends in `~/.config/kronk/backends/`
- **Windows:** Backends in `%APPDATA%\kronk\backends\`
- Version tracking in `~/.config/kronk/backend_registry.toml` (Linux/macOS) or `%APPDATA%\kronk\backend_registry.toml` (Windows)

### GPU Support

The installer detects your GPU and prompts you to select acceleration:

- **CUDA** (NVIDIA) — CUDA cores for faster inference
- **Vulkan** (AMD/Intel/NVIDIA) — Cross-platform GPU acceleration
- **Metal** (Apple Silicon) — macOS GPU acceleration
- **ROCm** (AMD) — AMD GPU support on Linux
- **CPU** — Fallback when no GPU is available

---

## Configuration

Kronk auto-generates a config on first run:

- **Windows:** `%APPDATA%\kronk\config\config.toml`
- **Linux:** `~/.config/kronk/config.toml`

```toml
[backends.llama_cpp]
path = "C:\\path\\to\\llama-server.exe"
health_check_url = "http://localhost:8080/health"

[models.my-model]
backend = "llama_cpp"
model = "bartowski/OmniCoder-8B-GGUF"
quant = "Q4_K_M"
profile = "coding"
enabled = true

[proxy]
host = "0.0.0.0"
port = 11434
idle_timeout_secs = 300
startup_timeout_secs = 120

[supervisor]
restart_policy = "always"
max_restarts = 10
restart_delay_ms = 3000
health_check_interval_ms = 5000
```

The `[models.*]` key (e.g. `my-model`) is the alias used by clients in `"model": "my-model"`. You can define multiple models. When `kronk serve` is running, request any enabled model and its backend will start automatically. Backend ports are auto-assigned — you don't need to configure them.

Model cards are stored in `~/.config/kronk/configs/<company>--<model>.toml` and contain quant info, context settings, and sampling presets.

### Directory Layout

```text
~/.config/kronk/
├── config.toml              Main configuration
├── profiles/              Sampling presets (editable)
│   ├── coding.toml
│   ├── chat.toml
│   ├── analysis.toml
│   └── creative.toml
├── configs/               Model cards
│   └── bartowski--OmniCoder-8B.toml
├── models/                  GGUF model files
│   └── bartowski/OmniCoder-8B/*.gguf
└── logs/                    Service logs
```

---

## How It Works

1. `kronk serve` starts an OpenAI-compatible API server on a single port (default 11434)
2. When a request arrives with `"model": "my-model"`, kronk looks up the config key in `[models.*]`
3. If the backend isn't running, kronk auto-assigns a free port, starts the backend with the right GGUF file, and waits for it to become healthy
4. The request is forwarded to the backend and the response is streamed back
5. After `idle_timeout_secs` of inactivity, the backend is shut down to free resources

### Service Integration

- **Windows:** Native Service Control Manager via the `windows-service` crate. `kronk service install` registers kronk as a Windows Service that auto-starts on boot. No NSSM or wrappers needed.
- **Linux:** Generates and manages systemd user units. `kronk service install` creates the unit file, enables it, and starts the service.

### Firewall (Windows)

`kronk service install` automatically adds an inbound firewall rule for port 11434. `kronk service remove` cleans it up.

---

## Architecture

```
kronk/
├── crates/
│   ├── kronk-core/      # Config, process supervisor, platform abstraction
│   ├── kronk-cli/       # CLI binary (clap)
│   └── kronk-mock/      # Mock LLM backend for testing
├── installer/           # Inno Setup script (Windows installer)
├── modelcards/          # Community model cards
├── .github/workflows/   # CI/CD release pipeline
└── README.md
```

---

## Building from Source

```bash
git clone https://github.com/danielcherubini/kronk.git
cd kronk
cargo build --release
```

The binary is at `target/release/kronk.exe` (Windows) or `target/release/kronk` (Linux).

---

## Roadmap

- **TUI Dashboard** — `kronk-tui` crate with ratatui
- **System tray** — Windows tray icon for quick service toggle
- **Tauri GUI** — Lightweight desktop frontend for non-CLI users

---

## Development

Kronk is built with modern Rust and follows these core crates:

- **kronk-core** — Core logic, process supervision, config management, platform abstractions
- **kronk-cli** — Command-line interface with clap, user prompts with inquire
- **kronk-mock** — Mock backend for testing and development

### Dependencies

Key dependencies include:

- `tokio` — Async runtime with process management
- `clap` — CLI parsing
- `serde` / `toml` — Configuration serialization
- `tracing` — Structured logging
- `reqwest` / `hf-hub` — HTTP client and HuggingFace integration
- `sysinfo` — System resource monitoring
- `indicatif` — Progress bars for downloads
- `directories` — Platform-specific config paths

