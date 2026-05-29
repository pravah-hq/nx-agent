from __future__ import annotations

import os
from pathlib import Path


class VlmClient:
    """Lazy-loaded Qwen3-VL on this machine (run agent + VLM together on GCP GPU VM)."""

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
            return (
                '{"action":"turn_right","pole_type":null,"stop_after":false,'
                '"reason":"dry run — no model loaded"}'
            )

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
