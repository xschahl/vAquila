# 🦅 vAquila

> **The Ollama developer experience, the vLLM production power.**

**vAquila** (accessible via the `vaq` command) is an open-source AI model inference manager. It combines the absolute simplicity of a CLI with the production performance of **vLLM** and the isolation of **Docker**, all with smart and automated GPU management.

## 🎯 The Problem
* **Ollama** is amazing for local testing, but its architecture shows its limits in production when handling multiple concurrent requests.
* **vLLM** is the undisputed king of production, but its deployment is a hassle (manual VRAM calculation, Docker volume management, *Out of Memory* crashes).

## ✨ The Solution: vAquila
vAquila orchestrates everything for you. Like an eagle soaring over your infrastructure, it analyzes your GPU state in real-time, calculates the perfect memory ratio, and deploys the vLLM Docker container invisibly and securely.

### Planned Features (Roadmap)
- [ ] **Auto-VRAM**: Automatic calculation of the `--gpu-memory-utilization` flag via NVML to prevent crashes.
- [ ] **One-Click Deployment**: Download and run models via a simple `vaq run <hf-model>` command.
- [ ] **Docker Orchestration**: Invisible management of containers, exposed ports, and Hugging Face cache.
- [ ] **Web UI**: A local dashboard to monitor active models and live GPU usage.

## 🚀 Concept / Quickstart (Under Development)

Instead of writing a complex, multi-line Docker command, simply run:

```bash
vaq run meta-llama/Llama-3-8B-Instruct
```

vAquila takes care of analyzing your NVIDIA cards, mounting the cache volumes and exposing the vLLM API on port 8000.

## 🛠️ Tech Stack (Prototype)
* **Language**: Python 3.10+
* **CLI**: Typer (or Click)
* **Orchestration**: Official Docker SDK for Python (docker)
* **Hardware**: pynvml for NVIDIA GPU monitoring
