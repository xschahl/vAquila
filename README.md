# 🦅 vAquila

<p align="center">
  <img src="src/vaquila/assets/preview.png" alt="vAquila" width="600px" />
</p>

> **The Ollama developer experience, the vLLM production power.**

**vAquila** (via the `vaq` command) is an open-source AI model inference manager. It combines the absolute simplicity of a CLI with the production performance of **vLLM** and the isolation of **Docker**, all with smart and automated GPU management.

---

## 🎯 The Problem & The Solution

- **The Problem:** **Ollama** is amazing for local testing, but its architecture shows limits in production. **vLLM** is the undisputed king of production performance, but its deployment is often a hassle (manual VRAM calculation, Docker volume management, _Out of Memory_ crashes).
- **The Solution:** vAquila orchestrates everything for you. Like an eagle soaring over your infrastructure, it analyzes your GPU state in real-time, calculates the perfect memory ratio, and deploys the vLLM Docker container invisibly and securely.

---

## 📚 Official Documentation

For the most up-to-date guide on how vAquila works and how to run it, please refer to the Docusaurus documentation:

### 👉 [View Documentation Here](https://xschahl.github.io/vAquila/docs)

**Quick Links:**

- 🚀 [Getting Started (Installation)](https://xschahl.github.io/vAquila/docs/getting-started)
- 💻 [CLI Reference (`vaq`)](https://xschahl.github.io/vAquila/docs/cli-reference)
- 🌐 [Web UI & Dashboard Guide](https://xschahl.github.io/vAquila/docs/web-ui)

---

## ✨ Key Features

- **Auto-VRAM**: Automatic calculation of the `--gpu-memory-utilization` flag via NVML to prevent crashes.
- **One-Click Deployment**: Download and run models via a simple `vaq run <hf-model>` command.
- **Advanced Model Compatibility**: Optional `trust_remote_code` support for repositories that require custom model code.
- **Docker Orchestration**: Invisible management of containers, exposed ports, and Hugging Face cache.
- **Web UI**: A local dashboard to manage models, containers, cache, and inference workflows.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **CLI**: Typer
- **Orchestration**: Official Docker SDK for Python
- **Hardware Monitoring**: `nvidia-ml-py` (NVML)
- **Inference Engine**: vLLM

---

_Built with ❤️ to make high-performance AI inference accessible to everyone._
