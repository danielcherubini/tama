<div align="center">

# Tama

> A local AI server with automatic backend management and a web-based control plane

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Build Status](https://img.shields.io/github/actions/workflow/status/danielcherubini/tama/ci.yml?label=CI&style=flat-square)](https://github.com/danielcherubini/tama/actions)

[Overview](#overview) • [Quick Start](#quick-start) • [Web UI](#web-ui) • [CLI Reference](#cli-reference) • [Configuration](#configuration) • [Architecture](#architecture)

</div>

![Tama Web UI](docs/screenshot.png)

## Overview

Tama is a local AI server written in Rust that provides an OpenAI-compatible API on a single port. It automatically manages backend lifecycles — starting models on demand, routing requests, and unloading idle models to save resources.

**Key features:**

- **OpenAI-compatible API** — Works with any client that supports the OpenAI API format
- **Automatic backend management** — Starts, routes, and unloads llama.cpp/ik_llama backends on demand
- **Web-based control plane** — Browser UI for managing models, viewing logs, and editing configuration
- **GPU acceleration** — Supports CUDA, Vulkan, Metal, and ROCm
- **Cross-platform** — Windows, Linux, and macOS support with native service integration
- **Model optimization** — Automatically detects VRAM and suggests optimal quantizations and context sizes

---

## Quick Start

### Installation

**Windows:** Download the installer from [Releases](https://github.com/danielcherubini/tama/releases), or:

```bash
cargo install --git https://github.com/danielcherubini/tama tama
```

**Linux (Debian/Ubuntu):**

```bash
sudo dpkg -i tama_*.deb
```

**Linux (Fedora/RHEL):**

```bash
sudo rpm -i tama-*.rpm
```

### Run Tama

```bash
tama service install
tama service start
```

> [!TIP]
> On Windows, Tama registers as a native Windows Service with firewall configuration. On Linux, it creates a systemd user unit.

---

## Web UI

Tama includes a web-based control plane for managing models, viewing logs, and editing configuration from your browser.

### Running the web UI

The web server starts automatically alongside the proxy when using `tama service start`.

For development or manual startup:

```bash
cargo run --package tama --features web-ui -- web --port 11435
```

Open [http://localhost:11435](http://localhost:11435) to access the dashboard.

> [!NOTE]
> The web UI proxies all `/tama/v1/` requests to the running Tama proxy (default `http://127.0.0.1:11434`).

### Dashboard features

- **Models page** — View installed models, pull new ones from HuggingFace, edit model configurations
- **Backends page** — Manage llama.cpp and ik_llama installations, update versions
- **Logs viewer** — Real-time log streaming with filtering
- **Config editor** — Edit configuration directly from the browser
- **Model status tiles** — See which models are running, their active backends, and job logs

---

## CLI Reference

### Server management

| Command | Description |
|---------|-------------|
| `tama serve` | Start the OpenAI-compatible API server (port 11434) |
| `tama status` | Show status of all servers and running models |
| `tama service install` | Install as a system service |
| `tama service start` | Start the service |
| `tama service stop` | Stop the service |
| `tama service restart` | Restart the service |
| `tama service remove` | Remove the service |

### Model management

| Command | Description |
|---------|-------------|
| `tama model pull <repo>` | Pull a model from HuggingFace |
| `tama model ls` | List installed models |
| `tama model create` | Create a model config from an installed model |
| `tama model enable <name>` | Enable a model for on-demand loading |
| `tama model disable <name>` | Disable a model |
| `tama model rm <model>` | Remove an installed model |
| `tama model scan` | Scan for untracked GGUF files |
| `tama model search <query>` | Search HuggingFace for GGUF models |
| `tama model update [model]` | Check for and download model updates |
| `tama model verify [model]` | Verify GGUF files against HuggingFace hashes |
| `tama model prune` | Remove orphaned GGUF files |

### Backend management

Tama manages LLM backend installations with automatic version tracking:

```bash
tama backend install llama_cpp     # Download pre-built llama.cpp binaries
tama backend install ik_llama      # Build from source
tama backend install llama_cpp --version b8407  # Specific version
tama backend install llama_cpp --build    # Force build from source
tama backend update <name>         # Update to latest version
tama backend list                  # List installed backends
tama backend remove <name>        # Remove a backend
tama backend check-updates         # Check for updates
```

### Server management

```bash
tama server ls                    # List all servers with status
tama server add <name> <cmd>      # Add a new server
tama server edit <name> <cmd>     # Edit an existing server
tama server rm <name>             # Remove a server
```

### Sampling profiles

```bash
tama profile list                  # List all available profiles
tama profile set <server> <name>  # Set a server's sampling profile
tama profile clear <server>        # Clear a server's sampling profile
```

### Configuration

| Command | Description |
|---------|-------------|
| `tama config show` | Print the current configuration |
| `tama config edit` | Open config file in editor |
| `tama config path` | Show the config file path |

### Utilities

| Command | Description |
|---------|-------------|
| `tama logs [name]` | View logs (defaults to proxy logs) |
| `tama run <name>` | Run a single backend for debugging |
| `tama bench [name]` | Benchmark model inference |
| `tama self-update` | Update Tama to the latest version |

---

## Configuration

Tama auto-generates a config on first run:

- **Windows:** `%APPDATA%\tama\config.toml`
- **Linux/macOS:** `~/.config/tama/config.toml`

```toml
[backends.llama_cpp]
path = "/path/to/llama-server"
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

### Directory layout

```
~/.config/tama/
├── config.toml              Main configuration
├── tama.db                   SQLite database (models, backends, pull history)
├── configs/                 Model cards with quant info and sampling presets
│   └── bartowski--OmniCoder-8B.toml
├── models/                  GGUF model files
│   └── bartowski/OmniCoder-8B/*.gguf
├── backends/                llama.cpp and ik_llama binaries
└── logs/                    Service logs
```

> [!NOTE]
> On first run after upgrading from kronk, Tama automatically migrates `~/.config/kronk` to `~/.config/tama`.

### GPU acceleration

The installer detects your GPU and offers these acceleration options:

- **CUDA** (NVIDIA) — Fast inference on NVIDIA GPUs
- **Vulkan** (AMD/Intel/NVIDIA) — Cross-platform GPU acceleration
- **Metal** (Apple Silicon) — Native macOS GPU acceleration
- **ROCm** (AMD) — AMD GPU support on Linux
- **CPU** — Fallback when no GPU is available

---

## Architecture

```
tama/
├── crates/
│   ├── tama-core/       # Config, process supervisor, proxy, platform abstraction
│   ├── tama-cli/        # CLI binary with clap
│   ├── tama-mock/       # Mock LLM backend for testing
│   └── tama-web/        # Leptos web control plane (WASM + SSR)
├── config/              # Configuration templates
├── docs/                # Documentation
├── installer/           # Windows Inno Setup script
└── modelcards/         # Community model cards
```

### Core components

- **tama-core** — Config management, process supervision, backend registry, proxy server, database
- **tama-cli** — Command-line interface with clap, interactive prompts with inquire
- **tama-web** — Leptos WASM frontend with real-time updates, SSR server for hosting
- **tama-mock** — Mock backend for testing and development

### How it works

1. `tama serve` (or `tama service start`) starts an OpenAI-compatible API server on port 11434
2. When a request arrives with `"model": "my-model"`, tama looks up the config
3. If the backend isn't running, tama auto-assigns a free port and starts it
4. The request is forwarded to the backend and the response is streamed back
5. After `idle_timeout_secs` of inactivity, the backend is shut down

---

## Building from source

```bash
git clone https://github.com/danielcherubini/tama.git
cd tama
cargo build --release
```

The binary is at `target/release/tama.exe` (Windows) or `target/release/tama` (Linux).

For development with the web UI:

```bash
# Install trunk for frontend builds
cargo install trunk

# Build and run with web features
cargo run --package tama --features web-ui -- web
```

---

## Roadmap

- **TUI Dashboard** — Terminal UI with ratatui for resource monitoring
- **System tray** — Quick service toggle from the system tray
- **Tauri GUI** — Lightweight desktop frontend for non-CLI users
