"""GPT-2 residual-stream logit lens, greedy generation, and a JSON CLI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

import torch
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel


class RequestError(ValueError):
    """An invalid request."""


def load_settings():
    # Only GPT-2 small is enabled. Actual layer/context counts come from its config.
    return {
        "allowed_models": {
            "gpt2": {
                "pretrained_id": "openai-community/gpt2",
                "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            }
        },
        "max_new_tokens": 1,
        "max_top_k": 50,
        "max_input_characters": 100000,
        "projection_chunk_size": 32,
    }


def validate_request(prompt, max_new_tokens, model, top_k, settings):
    if not isinstance(prompt, str) or not prompt:
        raise RequestError("prompt must be a non-empty string (spaces are preserved)")
    if len(prompt) > settings["max_input_characters"]:
        raise RequestError("prompt exceeds max_input_characters")
    if not isinstance(model, str) or model not in settings["allowed_models"]:
        raise RequestError(f"model must be one of {list(settings['allowed_models'])}")
    for name, value, minimum, maximum in (
        ("max_new_tokens", max_new_tokens, 0, settings["max_new_tokens"]),
        ("top_k", top_k, 1, settings["max_top_k"]),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise RequestError(f"{name} must be an integer in [{minimum}, {maximum}]")


class GPT2LogitLens:
    """Load one whitelisted pretrained model; reuse this object across requests.

    No random-weight fallback. A lock isolates temporary hooks and KV caches.
    CPU float32 is the default; device='mps' or 'auto' is also supported.
    """

    def __init__(self, model="gpt2", device="cpu", cache_dir=None,
                 local_files_only=False):
        self.settings = load_settings()
        if model not in self.settings["allowed_models"]:
            raise RequestError(f"model must be one of {list(self.settings['allowed_models'])}")
        spec = self.settings["allowed_models"][model]
        kwargs = dict(revision=spec["revision"], cache_dir=cache_dir,
                      local_files_only=local_files_only, token=False)
        config = AutoConfig.from_pretrained(spec["pretrained_id"], **kwargs)
        if config.model_type != "gpt2":
            raise ValueError("Only GPT-2 architecture is supported")
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in ("cpu", "mps"):
            raise ValueError("device must be cpu, mps, or auto")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable; use device='cpu'")
        self.device = torch.device(device)
        self.model_name = model
        self.model = GPT2LMHeadModel.from_pretrained(
            spec["pretrained_id"], config=config, use_safetensors=True,
            attn_implementation="eager", **kwargs,
        ).float().to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(spec["pretrained_id"], **kwargs)
        self.n_layers = self.model.config.n_layer
        if len(self.model.transformer.h) != self.n_layers:
            raise ValueError("Block count does not match model.config.n_layer")
        self.context_window = self.model.config.n_positions
        self._lock = RLock()

    def _forward(self, input_ids, past_key_values=None):
        """Capture raw post-block residuals BEFORE transformer.ln_f.

        HF hidden_states[-1] is already normalized: never use it as a raw block.
        Hooks also work with cached single-token decoding.
        """
        residuals = [None] * self.n_layers

        def capture(index):
            def hook(_module, _args, output):
                raw = output[0] if isinstance(output, tuple) else output
                residuals[index] = raw.detach()
            return hook

        handles = [block.register_forward_hook(capture(i))
                   for i, block in enumerate(self.model.transformer.h)]
        try:
            output = self.model(input_ids=input_ids, past_key_values=past_key_values,
                                use_cache=True, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if any(raw is None for raw in residuals):
            raise RuntimeError("Failed to capture every block")
        return output, residuals

    def _project(self, raw):
        return self.model.lm_head(self.model.transformer.ln_f(raw))

    def _decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    def analyze(self, prompt, max_new_tokens=1, model="gpt2", top_k=5):
        """Return JSON-safe predictions for every prompt and generated position.

        result.token_position=t always predicts token t+1. generation_step=0
        marks the row that produced the FIRST generated token (the last prompt
        position). The final token has reference_token=None; no future is known.
        max_new_tokens allows 0 or 1 tokenizer tokens, with a default of 1.
        top_k only controls lens output.
        """
        validate_request(prompt, max_new_tokens, model, top_k, self.settings)
        if model != self.model_name:
            raise RequestError(f"This engine loaded {self.model_name!r}, not {model!r}")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise RequestError("prompt must encode to at least one token")
        if len(ids) + max_new_tokens > self.context_window:
            raise RequestError(
                f"input tokens ({len(ids)}) + max_new_tokens ({max_new_tokens}) "
                f"must be <= model context window ({self.context_window})"
            )
        if top_k > self.model.config.vocab_size:
            raise RequestError("top_k exceeds model vocabulary size")

        with self._lock, torch.inference_mode():
            return self._analyze(prompt, ids, max_new_tokens, top_k)

    def _analyze(self, prompt, input_ids, max_new_tokens, top_k):
        ids = list(input_ids)
        input_count = len(ids)
        output, raw = self._forward(torch.tensor([ids], device=self.device))
        pieces = [[value] for value in raw]
        stop_reason = "max_new_tokens"
        for _step in range(max_new_tokens):
            # Generation uses native final logits, never top-k lens sampling.
            next_id = int(output.logits[0, -1].argmax().item())
            ids.append(next_id)
            # Feed the final generated token too, so its terminal lens exists.
            output, raw = self._forward(
                torch.tensor([[next_id]], device=self.device), output.past_key_values
            )
            for layer, value in enumerate(raw):
                pieces[layer].append(value)
            if next_id == self.tokenizer.eos_token_id:
                stop_reason = "eos_token"
                break

        tokens = [dict(position=p, token_id=token_id, text=self._decode([token_id]),
                       source="input" if p < input_count else "generated",
                       generation_step=None if p < input_count else p - input_count)
                  for p, token_id in enumerate(ids)]
        rows = []
        chunk_size = self.settings["projection_chunk_size"]
        for block_index, layer_pieces in enumerate(pieces):
            raw = torch.cat(layer_pieces, dim=1)[0]
            for start in range(0, len(ids), chunk_size):
                # Full-vocabulary softmax FIRST. Keep top-k probabilities as-is.
                logits = self._project(raw[start:start + chunk_size]).float()
                probs = logits.softmax(dim=-1)
                values, indices = probs.topk(top_k, dim=-1, sorted=True)
                values, indices = values.cpu().tolist(), indices.cpu().tolist()
                for offset, (top_probs, top_ids) in enumerate(zip(values, indices)):
                    p = start + offset
                    target_position = p + 1 if p + 1 < len(ids) else None
                    reference = None
                    if target_position is not None:
                        target_id = ids[target_position]
                        # Competition rank: equal logits share the same rank.
                        rank = 1 + int((logits[offset] > logits[offset, target_id]).sum().item())
                        reference = dict(token_id=target_id, token=self._decode([target_id]),
                                         rank=rank, probability=float(probs[offset, target_id].item()))
                    step = (target_position - input_count
                            if target_position is not None and target_position >= input_count
                            else None)
                    rows.append(dict(
                        block_index=block_index, token_position=p,
                        target_token_position=target_position, generation_step=step,
                        predictions=[dict(token_id=i, token=self._decode([i]),
                                          probability=v, rank=r + 1)
                                     for r, (i, v) in enumerate(zip(top_ids, top_probs))],
                        reference_token=reference,
                    ))
        rows.sort(key=lambda row: (row["token_position"], row["block_index"]))
        generated_ids = ids[input_count:]
        return dict(
            schema_version="1.1", request_id=str(uuid4()), status="success",
            model=self.model_name, method="logit_lens", is_mock=False,
            input_text=prompt, max_new_tokens=max_new_tokens, top_k=top_k,
            n_layers=self.n_layers, context_window=self.context_window,
            input_token_count=input_count, generated_token_count=len(generated_ids),
            generated_text=self._decode(generated_ids), output_text=self._decode(ids),
            stop_reason=stop_reason, tokens=tokens, results=rows,
        )


def read_request(path):
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RequestError(f"Cannot read request JSON: {exc}") from exc
    if not isinstance(request, dict):
        raise RequestError("request JSON must be an object")
    unknown = set(request) - {"prompt", "max_new_tokens", "model", "top_k"}
    if unknown:
        raise RequestError(f"Unknown request fields: {sorted(unknown)}")
    if "prompt" not in request:
        raise RequestError("request JSON must contain prompt")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path, help="UTF-8 request JSON file")
    source.add_argument("--prompt", help="Prompt text instead of a request file")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--model")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    try:
        request = dict(max_new_tokens=1, model="gpt2", top_k=5)
        request.update(read_request(args.request) if args.request else {"prompt": args.prompt})
        # Explicit flags override values from the file.
        for key in ("max_new_tokens", "model", "top_k"):
            if getattr(args, key) is not None:
                request[key] = getattr(args, key)
        validate_request(**request, settings=settings)
        engine = GPT2LogitLens(request["model"], args.device,
                              cache_dir=os.environ.get("ECE525_MODEL_CACHE"),
                              local_files_only=args.local_files_only)
        result = engine.analyze(**request)
    except RequestError as exc:
        parser.error(str(exc))
    text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        try:
            args.output.write_text(text, encoding="utf-8")
        except OSError as exc:
            parser.error(f"Cannot write output JSON: {exc}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
