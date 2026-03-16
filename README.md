# KRONK

High-performance, Rust-native cross-platform Service Orchestrator for local AI binaries.

## Features

- 🚀 Cross-platform (Linux/Windows)
- 🖥️ Windows Service integration  
- 📊 TUI dashboard
- 🔄 Auto-restart supervisor
- ⚙️ TOML config management

## Quick Start

```bash
cargo run --bin kronk-mock -- --port 8080
cargo run --bin kronk -- run --backend mock
```

## Documentation

See [SPEC.md](SPEC.md) for technical details.
