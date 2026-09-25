"""Fine-tune causal language models on a text corpus (the retrain step of remediation).

Models are saved with ``save_pretrained`` as safetensors: loading them never executes code,
unlike pickle. ``tiny_model`` builds a small model and tokenizer from scratch, entirely offline,
for tests and demos; real audits load an existing checkpoint.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from model_passport.llm.backends import HuggingFaceModel, ModelError, quiet_transformers


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 8
    max_length: int = 256
    weight_decay: float = 0.0
    seed: int = 0
    device: str | None = None


def _torch() -> Any:
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise ModelError("training needs the llm extra: pip install 'model-passport[llm]'") from exc
    return torch


def seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tiny_model(texts: Sequence[str], vocab_size: int = 2000, layers: int = 3, width: int = 128,
               seed: int = 0) -> HuggingFaceModel:  # fmt: skip
    """A small GPT-2 style model with a BPE tokenizer trained on ``texts``, fully offline."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: PLC0415
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast  # noqa: PLC0415

    quiet_transformers()
    seed_everything(seed)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))  # noqa: S106 - a token, not a password
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]  # stub lacks annotations
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(list(texts), trainer)
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<eos>",  # noqa: S106 - special tokens, not passwords
        unk_token="<unk>",  # noqa: S106
        pad_token="<eos>",  # noqa: S106
    )
    eos = wrapped.convert_tokens_to_ids("<eos>")
    config = GPT2Config(
        vocab_size=wrapped.vocab_size, n_positions=256, n_embd=width, n_layer=layers, n_head=4,
        bos_token_id=eos, eos_token_id=eos,
    )  # fmt: skip
    return HuggingFaceModel(GPT2LMHeadModel(config), wrapped, name="tiny-gpt2", device="cpu")


def finetune(
    model: HuggingFaceModel, texts: Sequence[str], settings: TrainSettings | None = None
) -> list[float]:
    """Train ``model`` in place with the causal LM objective; returns the mean loss per epoch."""
    torch = _torch()
    s = settings or TrainSettings()
    seed_everything(s.seed)
    network, tokenizer = model.model, model.tokenizer
    network.train()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=s.learning_rate, weight_decay=s.weight_decay
    )
    eos = tokenizer.eos_token or ""
    rows = [t + eos for t in texts]
    history = []
    for _ in range(s.epochs):
        order = list(range(len(rows)))
        random.shuffle(order)
        losses = []
        for start in range(0, len(order), s.batch_size):
            batch = [rows[i] for i in order[start : start + s.batch_size]]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=s.max_length
            ).to(model.device)
            labels = encoded["input_ids"].masked_fill(encoded["attention_mask"] == 0, -100)
            loss = network(**encoded, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    network.eval()
    return history


def save(model: HuggingFaceModel, directory: Path) -> Path:
    """Save weights (safetensors) and tokenizer so ``HuggingFaceModel.load`` can reload them."""
    directory.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(directory, safe_serialization=True)
    model.tokenizer.save_pretrained(directory)
    return directory
