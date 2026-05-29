# Split setup: agent locally, VLM on GCP GPU VM

```text
[Laptop]  python -m agent run --policy vlm
    |  HTTP POST /v1/complete  (JPEG crop + prompt)
    v
[GCP VM]  python -m agent.vlm_server  (Qwen3-VL-4B on GPU)
```

The laptop runs the agent loop, map math, and image crops. Only inference runs on the VM.

---

## A. GCP VM (VLM server)

### 1. GPU VM

- Ubuntu 22.04+ with NVIDIA driver (`nvidia-smi` works).
- Open firewall TCP **8788** (or your `VLM_SERVER_PORT`) to your laptop IP, or use SSH tunnel (below).

### 2. Install (on VM)

```bash
git clone <repo> nx-agent && cd nx-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements-vlm.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export HF_HOME=~/hf-cache
export VLM_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
# optional: export VLM_API_KEY=your-secret
```

You do **not** need panorama `data/` on the VM unless you also run the agent there.

### 3. Start server

```bash
source .venv/bin/activate
cd ~/nx-agent
python -m agent.vlm_server --host 0.0.0.0 --port 8788
```

First request downloads/loads the model (slow). Later requests are faster.

Health check from laptop:

```bash
curl http://VM_EXTERNAL_IP:8788/health
```

### 4. SSH tunnel (no public firewall)

On **laptop**:

```bash
ssh -L 8788:127.0.0.1:8788 user@VM_EXTERNAL_IP
```

On **VM** run server on `127.0.0.1:8788` or `0.0.0.0:8788`.  
On **laptop** set `VLM_REMOTE_URL=http://127.0.0.1:8788`.

---

## B. Local machine (agent)

### 1. Install

```bash
cd nx-agent
pip install -r requirements-agent.txt
```

Needs `data/metadata/` and `data/panoramas/` locally (see README Data Setup).

### 2. Point at remote VLM

**PowerShell:**

```powershell
$env:VLM_REMOTE_URL = "http://VM_EXTERNAL_IP:8788"
# if using SSH tunnel:
# $env:VLM_REMOTE_URL = "http://127.0.0.1:8788"
$env:VLM_API_KEY = "your-secret"   # only if set on server
```

**bash:**

```bash
export VLM_REMOTE_URL=http://VM_EXTERNAL_IP:8788
export VLM_API_KEY=your-secret     # optional
```

### 3. Run agent

```bash
python -m agent probe              # one remote inference
python -m agent run --policy vlm --max-steps 30
```

Dry run (no HTTP, no model):

```bash
$env:VLM_DRY_RUN = "1"   # PowerShell
python -m agent probe
```

### 4. Optional

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLM_REMOTE_URL` | — | If set, use remote server |
| `VLM_REMOTE_TIMEOUT` | `300` | Seconds per inference |
| `VLM_API_KEY` | — | Bearer token if server requires it |
| `VLM_TRACE_DIR` | — | Save prompts locally |

---

## C. Policies

| Command | Where GPU needed |
|---------|------------------|
| `run --policy stub` | Nowhere |
| `run --policy vlm` + `VLM_REMOTE_URL` | GCP VM only |
| `run --policy vlm` without `VLM_REMOTE_URL` | Same machine as agent |

---

## D. Troubleshooting

- **Connection refused** — server not running, wrong IP/port, or firewall.
- **401** — `VLM_API_KEY` mismatch between laptop and VM.
- **Timeout** — increase `VLM_REMOTE_TIMEOUT`; first load can take minutes.
- **CUDA OOM** — use a larger GPU or smaller model / quantization on VM.
