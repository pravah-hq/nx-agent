"""
HTTP VLM server — run on GCP GPU VM only.

  python -m agent.vlm_server --host 0.0.0.0 --port 8788

Local agent sets VLM_REMOTE_URL=http://<VM_IP>:8788
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from agent.model_client import LocalVlmClient


def create_app():
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

    app = FastAPI(title="nx-agent VLM server", version="0.1.0")
    engine = LocalVlmClient()
    api_key = os.environ.get("VLM_API_KEY", "")

    def check_auth(authorization: str | None) -> None:
        if not api_key:
            return
        if not authorization or authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model_id": engine.model_id, "dry_run": engine.dry_run}

    @app.post("/v1/complete")
    async def complete(
        prompt: str = Form(...),
        image: UploadFile = File(...),
        authorization: str | None = Header(default=None),
    ) -> dict:
        check_auth(authorization)
        suffix = Path(image.filename or "view.jpg").suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            content = await image.read()
            tmp.write(content)
            tmp.flush()
            tmp.close()
            text = engine.complete(prompt, Path(tmp.name))
            return {"text": text}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL HTTP server for nx-agent")
    parser.add_argument("--host", default=os.environ.get("VLM_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLM_SERVER_PORT", "8788")))
    args = parser.parse_args(argv)

    import uvicorn

    print(f"VLM server listening on http://{args.host}:{args.port}", flush=True)
    print(f"On your laptop: set VLM_REMOTE_URL=http://<VM_EXTERNAL_IP>:{args.port}", flush=True)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
