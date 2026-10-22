# Troubleshooting & Engineering Runbook: Edge Gemma 3 Deployment

This runbook documents the exact failure modes, diagnostic commands, and resolutions encountered when deploying Gemma 3 4B on a 4GB Raspberry Pi.

---

## 1. Out-of-Memory (OOM) Termination

### Symptom
Running `llama-cli` terminates immediately during the weight initialization phase:
```text
Loading model... |Killed
```

### Diagnostic
Checking the Linux kernel ring buffer:
```bash
dmesg -T | grep -i -E "oom|killed"
```
Output confirms invocation of the Linux `oom-killer` targeting `llama-cli` due to exhausted anonymous memory.

### Root Cause
1. `llama.cpp` allocates context buffers based on `-c` (defaulting to 4096 or 8192).
2. Model weights in memory: ~2.45 GiB.
3. System services (systemd, networking, SSH): ~495 MiB.
4. Total free RAM was ~1.0 GiB. Initializing a 4096-token KV cache on 4 threads requires >1.2 GiB of auxiliary buffers, exceeding the remaining memory pool.

### Resolution
Constrain the context allocation and token budget explicitly:
```bash
./build/bin/llama-mtmd-cli \
    -m ~/models/gemma-3-4b-it-Q4_K_M.gguf \
    --mmproj ~/models/mmproj-gemma-3-4b-f16.gguf \
    --image ~/datasets/input.png \
    -n 128 \
    -c 1024 \
    -t 4
```

---

## 2. Zero-Byte Storage Exhaustion (`No usable temporary directory found`)

### Symptom
Python scripts fail to launch with:
```text
FileNotFoundError: [Errno 2] No usable temporary directory found in ['/tmp', '/var/tmp', '/usr/tmp', '/home/ubuntu']
```

### Diagnostic
```bash
df -h /
```
Output:
```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/mmcblk0p2   14G   14G     0 100% /
```

### Root Cause
Zero available storage bytes prevents Python's `tempfile` module from creating a test file upon interpreter initialization. The storage consumption was tracked down using:
```bash
du -h -d 1 ~ 2>/dev/null | sort -rh | head -10
du -sh ~/.cache/* 2>/dev/null | sort -rh | head -10
```
Key culprits:
1. `~/.cache/pip`: Accumulated **5.7 GB** of cached `.whl` files.
2. `~/llama.cpp/build/CMakeFiles`: Accumulated **~1.0 GB** of object files from the CMake build.

### Resolution
Run the automated cleanup script:
```bash
./scripts/system_cleanup.sh
```
Or manually execute:
```bash
rm -rf ~/.cache/pip
rm -rf ~/llama.cpp/build/CMakeFiles/
sudo apt-get clean
sudo journalctl --vacuum-size=100M
```

---

## 3. Deprecated Multimodal CLI Tool (`llama-llava-cli`)

### Symptom
When executing multimodal vision commands:
```text
llama-llava-cli is deprecated. Please migrate to llama-mtmd-cli.
```

### Resolution
Migrate all automation scripts from `llama-llava-cli` to `llama-mtmd-cli`. The syntax remains compatible, but `llama-mtmd-cli` features unified multi-modal projector handling and improved token streaming.

---

## 4. Hardware Undervoltage Throttling

### Symptom
Desktop GUI notification:
```text
Low voltage warning — Please check your power supply
```
Inference throughput degrades from ~3.4 tokens/second down to ~1.8 tokens/second.

### Diagnostic
Query the Raspberry Pi power management IC via `vcgencmd`:
```bash
vcgencmd get_throttled
```
A return value of `0x50000` or `0x50005` confirms that under-voltage occurred and frequency capping is currently active.

### Root Cause
4-core SIMD NEON execution draws sudden transient current spikes up to 2.8A. Standard 5V phone chargers drop to ~4.6V under load, triggering the hardware protection circuit.

### Resolution
Power the board with the official Raspberry Pi 15W (5V/3A) or 27W (5.1V/5A) USB-C power supply using heavy-gauge wiring.
