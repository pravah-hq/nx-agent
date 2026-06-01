"""
Qwen3-VL wrapper for map + street image prompts.

Environment:
  VLM_MODEL_ID, VLM_MAX_NEW_TOKENS, VLM_DRY_RUN=1 (keyword-based fake JSON, no GPU)
  VLM_DEVICE_MAP, VLM_ATTN_IMPLEMENTATION
"""

from __future__ import annotations

import os
from pathlib import Path


class VlmClient:
    """Lazy-loads Hugging Face Qwen3-VL; use one instance per VlmPolicy."""

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
        return self.complete_images(prompt, [image_path])

    def complete_images(self, prompt: str, image_paths: list[Path]) -> str:
        """
        Multi-image chat: image order must match prompt (map first, street second).

        dry_run branches on prompt keywords — update when adding new prompt types.
        """
        if self.dry_run:
            lower = prompt.lower()
            if "pole_type_definitions" in lower and "your answer controls" in lower:
                return (
                    '{"pole_in_clear_view":true,'
                    '"identifiable_pole_type":"lamp_post",'
                    '"confirmed_target_pole_id":null,'
                    '"reason":"dry run — target visible"}'
                )
            if "pole_in_clear_view" in lower and "pole_type_definitions" not in lower:
                return '{"pole_in_clear_view":false,"reason":"dry run clear view"}'
            if "view_clear" in lower and "pole_type_definitions" not in lower:
                return '{"view_clear":true,"reason":"dry run visibility"}'
            if "pole_type_definitions" in lower or "classify the target pole" in lower:
                return (
                    '{"classified_pole_id":"POLE_000057",'
                    '"pole_type":"distribution_transformer",'
                    '"confidence":"low","reason":"dry run type pass"}'
                )
            if "two images" in lower or "image 1" in lower:
                return (
                    '{"action":"turn_right","target_pano_id":null,'
                    '"reason":"dry run dual navigation"}'
                )
            return (
                '{"action":"turn_right","target_pano_id":null,'
                '"reason":"dry run — no model loaded"}'
            )

        for path in image_paths:
            if not path.is_file():
                raise FileNotFoundError(f"VLM image not found: {path}")

        self._load()
        assert self._model is not None
        assert self._processor is not None

        content: list[dict] = []
        for path in image_paths:
            content.append({"type": "image", "image": str(path.resolve())})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

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
