from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VlmClientProtocol(Protocol):
    dry_run: bool

    def complete(self, prompt: str, image_path: Path) -> str: ...


class LocalVlmClient:
    """Load Qwen3-VL on this machine (use on GCP GPU VM)."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        max_new_tokens: int | None = None,
        dry_run: bool | None = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
        self.max_new_tokens = max_new_tokens or int(os.environ.get("VLM_MAX_NEW_TOKENS", "256"))
        env_dry = os.environ.get("VLM_DRY_RUN", "").lower() in {"1", "true", "yes"}
        self.dry_run = dry_run if dry_run is not None else env_dry
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        kwargs: dict = {
            "dtype": "auto",
            "device_map": os.environ.get("VLM_DEVICE_MAP", "auto"),
        }
        attn = os.environ.get("VLM_ATTN_IMPLEMENTATION")
        if attn:
            kwargs["attn_implementation"] = attn

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model.eval()
        if torch.cuda.is_available():
            print(f"VLM loaded on CUDA ({torch.cuda.get_device_name(0)})", flush=True)
        else:
            print("VLM loaded (no CUDA detected — inference will be slow)", flush=True)

    def complete(self, prompt: str, image_path: Path) -> str:
        if self.dry_run:
            return _dry_run_response()

        if not image_path.is_file():
            raise FileNotFoundError(f"VLM image not found: {image_path}")

        self._load()
        assert self._model is not None
        assert self._processor is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        import torch

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        decoded = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


class RemoteVlmClient:
    """HTTP client: agent runs locally, inference on GCP VLM server."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout_s: int | None = None,
        dry_run: bool | None = None,
    ) -> None:
        remote = base_url or os.environ.get("VLM_REMOTE_URL", "").strip()
        if not remote:
            raise ValueError("VLM_REMOTE_URL is required for RemoteVlmClient")
        self.base_url = remote.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("VLM_API_KEY", "")
        self.timeout_s = timeout_s or int(os.environ.get("VLM_REMOTE_TIMEOUT", "300"))
        env_dry = os.environ.get("VLM_DRY_RUN", "").lower() in {"1", "true", "yes"}
        self.dry_run = dry_run if dry_run is not None else env_dry

    def health(self) -> dict:
        return self._get_json("/health")

    def complete(self, prompt: str, image_path: Path) -> str:
        if self.dry_run:
            return _dry_run_response()

        if not image_path.is_file():
            raise FileNotFoundError(f"VLM image not found: {image_path}")

        try:
            import requests
        except ImportError as err:
            raise ImportError(
                "Remote VLM client needs `requests`. Install: pip install requests"
            ) from err

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with image_path.open("rb") as handle:
            response = requests.post(
                f"{self.base_url}/v1/complete",
                data={"prompt": prompt},
                files={"image": (image_path.name, handle, "image/jpeg")},
                headers=headers,
                timeout=self.timeout_s,
            )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"Invalid VLM server response: {payload}")
        return text.strip()

    def _get_json(self, path: str) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(f"{self.base_url}{path}", headers=headers)
        try:
            with urlopen(request, timeout=min(self.timeout_s, 30)) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM server HTTP {err.code}: {body}") from err
        except URLError as err:
            raise RuntimeError(f"VLM server unreachable at {self.base_url}: {err}") from err


def _dry_run_response() -> str:
    return (
        '{"action":"turn_right","pole_type":null,"stop_after":false,'
        '"reason":"dry run — no model call"}'
    )


def build_vlm_client(**kwargs) -> VlmClientProtocol:
    """Local GPU if VLM_REMOTE_URL unset; else HTTP to remote server."""
    if os.environ.get("VLM_REMOTE_URL", "").strip():
        return RemoteVlmClient(**kwargs)
    return LocalVlmClient(**kwargs)


# Back-compat alias
VlmClient = LocalVlmClient
