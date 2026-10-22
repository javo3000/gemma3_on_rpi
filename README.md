# Gemma 3 4B on Raspberry Pi: Edge Multimodal Clinical OCR Pipeline

[![Target](https://img.shields.io/badge/Hardware-Raspberry%20Pi%204%2F5%20(4GB)-red.svg)](https://www.raspberrypi.com/)
[![Engine](https://img.shields.io/badge/Engine-llama.cpp%20(ARM64)-blue.svg)](https://github.com/ggerganov/llama.cpp)
[![Model](https://img.shields.io/badge/Model-Gemma%203%204B%20Instruct%20(GGUF)-yellow.svg)](https://huggingface.co/google/gemma-3-4b-it)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

An end-to-end systems implementation, memory management framework, and benchmarking pipeline for deploying Google's **Gemma 3 4B** multimodal model on resource-constrained **Raspberry Pi (4GB RAM)** hardware without internet connectivity.

Developed as part of the **MedVi** project to digitize handwritten clinical records from the African Medical Records (AMR) benchmark under zero-cloud, rural healthcare facility conditions.

---

## 📋 Architectural Overview

```
                      +---------------------------------------+
                      |       Clinical Document Image         |
                      |          (AMR_001_HTR.png)            |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         llama-mtmd-cli Engine         |
                      |   (llama.cpp compiled with NEON)      |
                      +-------------------+-------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
+-------------------------------+                   +-------------------------------+
|     Gemma 3 4B Q4_K_M GGUF     |                   |    mmproj-f16 Vision Projector|
|            (~2.4 GB)          |                   |            (~300 MB)          |
+---------------+---------------+                   +---------------+---------------+
                |                                                   |
                +-------------------------+-------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     Memory & Context Constraint       |
                      |     (-c 1024, -n 128, -t 4 threads)   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Digitized Text Transcription     |
                      |         (~3.4 tokens/second)          |
                      +---------------------------------------+
```

---

## ⚡ Performance Benchmark Summary

Evaluated on 62 handwritten clinical notes from the African Medical Records (AMR) OCR benchmark:

| Metric | Cloud Baseline (NVIDIA H200 AWQ) | Edge Target (Raspberry Pi 4B GGUF) |
| :--- | :--- | :--- |
| **Quantization** | AWQ 4-Bit (`W4A16`) | GGUF `Q4_K_M` + F16 Projector |
| **Working Set Memory** | ~3.5 GB VRAM | ~3.2 GB RAM (within 4GB limit) |
| **Generation Throughput** | 315.8 tok/s | **3.40 tok/s** |
| **Character Error Rate (CER)**| 0.3382 | **0.3410** (99.1% parity preserved) |
| **Word Error Rate (WER)** | 0.5992 | **0.6045** |
| **Average Document Latency** | ~0.40s | **~37.6s** |
| **Power Consumption** | ~350W - 700W | **~5.5W - 7.8W** |

---

## 🛠️ Hardware Requirements

* **Board:** Raspberry Pi 4 Model B (4GB or 8GB) or Raspberry Pi 5.
* **Storage:** ≥16GB MicroSD card (High-speed Class 10 / A2 rated recommended).
* **Power Supply:** Official 5V / 3A (15W+) USB-C power adapter (prevents underclocking under 4-core AVX/NEON load).
* **Cooling:** Active fan cooler or passive aluminum case heat sink.

---

## 🚀 Quickstart & Setup

### 1. Clone Repository & Setup System Dependencies
```bash
git clone https://github.com/javo3000/gemma3_on_rpi.git
cd gemma3_on_rpi

# Make helper scripts executable
chmod +x scripts/*.sh scripts/*.py

# Run automated system setup (compiles llama.cpp, configures swap)
./scripts/setup_rpi.sh
```

### 2. Download Model & Vision Projector Weights
```bash
mkdir -p ~/models

# Download Gemma 3 4B Q4_K_M quantized GGUF (~2.4 GB)
curl -L -o ~/models/gemma-3-4b-it-Q4_K_M.gguf \
  https://huggingface.co/google/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf

# Download Multimodal Vision Projector F16 (~300 MB)
curl -L -o ~/models/mmproj-gemma-3-4b-f16.gguf \
  https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf
```

### 3. Run Multimodal Inference on a Document Image
```bash
./scripts/run_inference.sh \
  --image ./samples/AMR_001_HTR.png \
  --output ./predictions/AMR_001_PREDICTION.txt
```

---

## 📂 Repository Structure

```
gemma3-rpi-deployment/
├── README.md                      # Primary project overview and benchmarks
├── docs/
│   └── troubleshooting_runbook.md # Post-mortem on OOM, undervoltage, and storage failures
├── scripts/
│   ├── setup_rpi.sh               # Environment provisioning and llama.cpp compilation
│   ├── run_inference.sh           # Parameter-tuned llama-mtmd-cli execution wrapper
│   ├── eval_pi.py                 # CER/WER batch evaluation against ground truth
│   └── system_cleanup.sh          # Pip cache and build artifact storage scrubber
└── samples/
    └── sample_transcription.txt   # Example transcription output
```

---

## 🔧 Critical Systems Engineering Insights

### Preventing OOM Terminations (`Loading model... | Killed`)
By default, `llama.cpp` attempts to allocate 4096 tokens of KV cache context. On a 4GB Pi with 2.4GB weights, this triggers the Linux OOM killer. We restrict the runtime budget:
* Context window: `-c 1024`
* Generation limit: `-n 128`
* Thread pin: `-t 4`

### Managing Storage Ceilings on 16GB MicroSD Cards
Flash storage exhaustion stops Python temp directories from functioning (`FileNotFoundError: [Errno 2] No usable temporary directory found`). Our automated scrubber recovers over 6GB of space by:
* Purging pip wheels: `rm -rf ~/.cache/pip`
* Removing intermediate CMake `.o` objects: `rm -rf ~/llama.cpp/build/CMakeFiles/`
* Rotating system journal logs: `sudo journalctl --vacuum-size=100M`

### Hardware Undervoltage Throttling
Inference with NEON SIMD instructions on 4 cores draws sharp current peaks. Cheap phone chargers cause rail voltage to drop below 4.65V, triggering CPU clock underclocking from 1.5GHz down to 600MHz. An official 15W+ adapter is mandatory for full 3.4 tok/s throughput.

---

## 📄 Citation & Attribution

If you use this work or reference these benchmarks in your research or applications, please cite:

```bibtex
@misc{avo2026gemma3rpi,
  author = {Avo, J.},
  title = {Edge Multimodal LLM Deployment on Resource-Constrained Hardware: Gemma 3 4B on Raspberry Pi for Clinical OCR},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/javo3000/gemma3_on_rpi}}
}
```
