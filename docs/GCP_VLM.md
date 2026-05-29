# Run agent + VLM on a GCP GPU VM

Everything runs on the **same VM**: agent loop, panorama crops, and Qwen3-VL-4B-Instruct.

## 1. VM and data

- GPU VM with `nvidia-smi` working.
- Clone or copy the repo to `~/nx-agent`.
- Copy `data/` (metadata + panoramas) onto the VM:

```bash
# From laptop (use VM external IP or IAP SSH):
scp -r data/ user@VM_IP:~/nx-agent/
```

## 2. Python venv

```bash
cd ~/nx-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-agent.txt
pip install -r requirements-vlm.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 3. Environment

```bash
export HF_HOME=~/hf-cache
export VLM_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
export VLM_DEVICE_MAP=auto
export VLM_MAX_NEW_TOKENS=256
# optional: export VLM_TRACE_DIR=~/nx-agent/vlm-traces
```

Dry run (no GPU load):

```bash
export VLM_DRY_RUN=1
python -m agent probe
unset VLM_DRY_RUN
```

## 4. Commands (on the VM)

```bash
source .venv/bin/activate
cd ~/nx-agent

python -m agent state
python -m agent probe
python -m agent probe   # shows map image path, phase, action
python -m agent run --policy vlm --max-steps 50
python -m agent run --policy stub --max-steps 200   # no GPU, fast planner
```

## 5. Optional: web UI on the VM

```bash
npm install
npm run backend &
export VITE_API_BASE=http://127.0.0.1:8787
npm run frontend -- --host 0.0.0.0 --port 5177
```

Use SSH port-forward to view from laptop: `ssh -L 5177:127.0.0.1:5177 user@VM_IP`.

## Troubleshooting

- **CUDA OOM** — smaller GPU, fewer steps, or quantized loading.
- **Missing data** — run `npm run check-data` or ensure `data/panoramas/` paths match metadata.
- **Slow** — each `--policy vlm` step runs one full model inference; start with `probe` then low `--max-steps`.
