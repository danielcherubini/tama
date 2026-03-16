# Technical Specification: Project **KRONK**

**The Heavy-Lifting Henchman for Local AI**

---

## 1. Executive Summary

**KRONK** is a high-performance, Rust-native Windows Service Orchestrator designed to interface with local AI binaries like `ik_llama` and `llama.cpp`. Unlike "black-box" wrappers, Kronk provides a system-level management layer that allows users to run models as persistent Windows Services with a power-user TUI (Terminal User Interface) and optional GUI.

The goal is to bring the "Ollama experience" to the raw power of specialized Windows binaries, specifically targeting the optimization flags found in `ik_llama`.

---

## 2. The "Kronk" Persona & UX

To differentiate from corporate tools, the interface utilizes a playful "Lever Lab" theme:

- **The CLI:** Command structures follow the "Pull the lever!" motif (e.g., `kronk pull <model>`).
- **The TUI:** A "War Room" dashboard showing VRAM, tokens/sec, and real-time logs using `ratatui`.
- **The Tray:** A small golden lever icon in the Windows System Tray for quick service toggles.

---

## 3. Technical Architecture

### **Core Stack**

- **Language:** Rust (Stable).
- **Async Runtime:** `tokio` for non-blocking I/O and process supervision.
- **Interface:**
  - **CLI:** `clap` (v4+) for command parsing.
  - **TUI:** `ratatui` for the dashboard and log streaming.
  - **GUI:** `Tauri` (v2) for a lightweight Windows frontend.

### **Windows Integration**

- **`windows-service`:** For native SCM (Service Control Manager) registration.
- **`tray-icon`:** For system-tray integration.
- **`tokio::process`:** To spawn and supervise `.exe` child processes, capturing `stdout/stderr`.

---

## 4. Feature Requirements

### **A. Service Orchestration**

- **Service Wrapping:** Ability to wrap `ik_llama.exe` into a persistent Windows service that survives user logout.
- **The "Wrong Lever" Protocol:** Auto-restart logic that triggers if the backend process crashes or hangs.
- **Pre-boot Execution:** Option to start the LLM backend at system boot.

### **B. `ik_llama` Specialization**

- Native support for specialized flags: `--CUDA_GRAPH_OPT=1`, `-sm graph`, `-khad`, and custom quants.
- Configuration profiles (TOML) for switching between "Speed" and "Precision" settings.

### **C. The "War Room" TUI**

- **Log Tailer:** Real-time streaming of model output without locking the terminal.
- **Health Monitor:** Live visualization of VRAM usage, CPU spikes, and temperature.

---

## 5. Senior Engineer Query

> **Copy/Paste this to your developer:**
>
> "I want to build a tool called **Kronk**. It’s a local AI service manager and process supervisor for Windows, written in **Rust**.
>
> **The Goal:** Create a system-level orchestrator that acts as a frontend for `ik_llama` and `llama.cpp`. It needs to manage these binaries as native Windows Services so they run in the background without keeping a terminal open.
>
> **The Challenge:** I need a highly resilient process supervisor using `tokio` that can handle IPC/Pipe redirection for logs and performance metrics. It should have a 'Pull the Lever' themed CLI using `clap` and a dashboard TUI using `ratatui`.
>
> **Architecture Requirements:**
>
> 1. **Monitor Thread:** A background worker to check child process health and VRAM usage.
> 2. **Service Wrapper:** Integration with `windows-service` so users can run `kronk service --install` to register the model as a boot-time service.
> 3. **Specialized Backend Support:** Hard-coded optimization profiles for `ik_llama` flags (e.g., CUDA Graph optimizations).
> 4. **Config Management:** Use `serde` with TOML to handle model paths and argument sets.
>
> Can you draft the initial workspace structure, the `Cargo.toml` dependencies, and a prototype of the `ServiceControl` logic for spawning the child process?"
