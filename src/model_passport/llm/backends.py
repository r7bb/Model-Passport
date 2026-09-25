"""Language models under audit, behind two small interfaces.

``ScoringModel`` returns per-token log-probabilities with character offsets. It powers the
likelihood attacks and needs white-box or log-prob API access: a Hugging Face model, or an
OpenAI-compatible server that echoes prompt log-probs (vLLM, TGI, and many hosted providers).

``GenerationModel`` only completes text. Hosted chat models (Anthropic, OpenAI) usually expose
no prompt log-probs, so they are audited by extraction probing instead.

``load_model`` builds either from a spec string::

    hf:EleutherAI/pythia-160m          Hugging Face Hub id (or a local directory path)
    openai-compatible:http://host/v1#model-name
    anthropic:claude-sonnet-5
    openai:gpt-5
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

HTTP_TIMEOUT = 60.0
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1"


class ModelError(RuntimeError):
    """The model could not be loaded or queried."""


@dataclass(frozen=True)
class Scored:
    """Token log-probabilities of one text.

    ``logprobs[i]`` is log P(token i | tokens before it); the first token has no context and is
    NaN. ``offsets[i]`` is the token's (start, end) character span in the text.
    """

    offsets: list[tuple[int, int]]
    logprobs: np.ndarray


@runtime_checkable
class ScoringModel(Protocol):
    name: str

    def score(self, texts: Sequence[str]) -> list[Scored]: ...


@runtime_checkable
class GenerationModel(Protocol):
    name: str

    def complete(
        self, prompt: str, *, n: int = 1, max_tokens: int = 32, temperature: float = 1.0
    ) -> list[str]: ...


# --- Hugging Face ------------------------------------------------------------------------------


def quiet_transformers() -> None:
    """Silence Transformers' progress bars and advisory warnings in CLI and report output."""
    from transformers.utils import logging  # noqa: PLC0415 - optional dependency

    logging.set_verbosity_error()
    logging.disable_progress_bar()


def _device(requested: str | None) -> str:
    import torch  # noqa: PLC0415 - optional dependency

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HuggingFaceModel:
    """A causal language model from the Hugging Face Hub or a local directory."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        name: str,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.device = _device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.name = name
        self.batch_size = batch_size

    @classmethod
    def load(
        cls, source: str | Path, device: str | None = None, batch_size: int = 8
    ) -> HuggingFaceModel:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise ModelError("install the llm extra: pip install 'model-passport[llm]'") from exc
        quiet_transformers()
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(source))
            model = AutoModelForCausalLM.from_pretrained(str(source))
        except (OSError, ValueError) as exc:
            raise ModelError(f"cannot load {source}: {exc}") from exc
        return cls(model, tokenizer, str(source), device, batch_size)

    def score(self, texts: Sequence[str]) -> list[Scored]:
        import torch  # noqa: PLC0415 - optional dependency

        results: list[Scored] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch, return_tensors="pt", padding=True, return_offsets_mapping=True
            )
            ids = encoded["input_ids"].to(self.device)
            mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
            picked = logprobs.gather(2, ids[:, 1:, None])[..., 0].cpu().numpy()
            for row, text_mask in enumerate(encoded["attention_mask"]):
                length = int(text_mask.sum())
                offsets = [tuple(o) for o in encoded["offset_mapping"][row][:length].tolist()]
                values = np.concatenate([[np.nan], picked[row][: length - 1]])
                results.append(Scored(offsets, values))
        return results

    def complete(
        self, prompt: str, *, n: int = 1, max_tokens: int = 32, temperature: float = 1.0
    ) -> list[str]:
        import torch  # noqa: PLC0415 - optional dependency

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        return [
            self.tokenizer.decode(seq[prompt_length:], skip_special_tokens=True) for seq in output
        ]


# --- HTTP APIs ---------------------------------------------------------------------------------


def _http() -> Any:
    try:
        import httpx  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise ModelError("API models need httpx: pip install 'model-passport[llm]'") from exc
    return httpx


def _key(env: str) -> str:
    value = os.environ.get(env)
    if not value:
        raise ModelError(f"set {env} to audit this API model")
    return value


class OpenAICompatibleModel:
    """An OpenAI-style completions server that echoes prompt log-probs (vLLM, TGI, ...)."""

    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"{model}@{self.base_url}"
        self.api_key = os.environ.get(api_key_env, "")

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        httpx = _http()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = httpx.post(
                f"{self.base_url}{path}", json=body, headers=headers, timeout=HTTP_TIMEOUT
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelError(f"{self.name}: {exc}") from exc
        data: dict[str, Any] = response.json()
        return data

    def score(self, texts: Sequence[str]) -> list[Scored]:
        results = []
        for text in texts:
            data = self._post(
                "/completions",
                {"model": self.model, "prompt": text, "max_tokens": 1, "echo": True,
                 "logprobs": 0, "temperature": 0},
            )  # fmt: skip
            logprobs = data["choices"][0].get("logprobs") or {}
            tokens = logprobs.get("tokens") or []
            starts = logprobs.get("text_offset") or []
            values = logprobs.get("token_logprobs") or []
            if not tokens or len(starts) != len(tokens):
                raise ModelError(f"{self.name} did not echo prompt log-probs")
            # The response may include one generated token after the prompt; drop it.
            keep = [i for i, s in enumerate(starts) if s < len(text)]
            offsets = [(starts[i], min(starts[i] + len(tokens[i]), len(text))) for i in keep]
            lp = np.array([np.nan if values[i] is None else values[i] for i in keep], dtype=float)
            results.append(Scored(offsets, lp))
        return results

    def complete(
        self, prompt: str, *, n: int = 1, max_tokens: int = 32, temperature: float = 1.0
    ) -> list[str]:
        data = self._post(
            "/completions",
            {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "n": n,
             "temperature": temperature},
        )  # fmt: skip
        return [str(choice.get("text", "")) for choice in data["choices"]]


class AnthropicModel:
    """Claude through the Messages API. Generation only: no prompt log-probs are exposed."""

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL) -> None:
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(
        self, prompt: str, *, n: int = 1, max_tokens: int = 32, temperature: float = 1.0
    ) -> list[str]:
        httpx = _http()
        headers = {
            "x-api-key": _key("ANTHROPIC_API_KEY"),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": min(temperature, 1.0),
            "messages": [{"role": "user", "content": prompt}],
        }
        outputs = []
        for _ in range(n):
            try:
                response = httpx.post(
                    ANTHROPIC_URL, json=body, headers=headers, timeout=HTTP_TIMEOUT
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ModelError(f"{self.name}: {exc}") from exc
            blocks = response.json().get("content", [])
            outputs.append("".join(b.get("text", "") for b in blocks if b.get("type") == "text"))
        return outputs


class OpenAIChatModel:
    """An OpenAI chat model. Generation only: chat responses carry no prompt log-probs."""

    def __init__(self, model: str, base_url: str = OPENAI_URL) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = f"openai:{model}"

    def complete(
        self, prompt: str, *, n: int = 1, max_tokens: int = 32, temperature: float = 1.0
    ) -> list[str]:
        httpx = _http()
        headers = {"Authorization": f"Bearer {_key('OPENAI_API_KEY')}"}
        body = {
            "model": self.model,
            "n": n,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions", json=body, headers=headers,
                timeout=HTTP_TIMEOUT,
            )  # fmt: skip
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelError(f"{self.name}: {exc}") from exc
        return [c["message"].get("content") or "" for c in response.json()["choices"]]


def load_model(spec: str, device: str | None = None) -> ScoringModel | GenerationModel:
    """A model from a spec string (see the module docstring)."""
    kind, _, rest = spec.partition(":")
    if kind == "hf":
        return HuggingFaceModel.load(rest, device)
    if kind == "openai-compatible":
        url, _, model = rest.rpartition("#")
        if not url or not model:
            raise ModelError("use openai-compatible:<base url>#<model name>")
        return OpenAICompatibleModel(url, model)
    if kind == "anthropic":
        return AnthropicModel(rest or DEFAULT_ANTHROPIC_MODEL)
    if kind == "openai":
        return OpenAIChatModel(rest)
    if Path(spec).is_dir():
        return HuggingFaceModel.load(spec, device)
    raise ModelError(f"unknown model spec {spec!r}; see model_passport.llm.backends")


def logmeanexp(values: Sequence[float]) -> float:
    """log(mean(exp(values))), computed stably."""
    array = np.asarray(values, dtype=float)
    top = float(array.max())
    return top + math.log(float(np.mean(np.exp(array - top))))
