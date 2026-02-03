from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class LocalConfig:
    backend: str  # "transformers"
    device: str  # "cpu" | "cuda"
    seed: int
    checkpoint_dir: str
    max_seq_length: int
    load_in_4bit: bool

    @classmethod
    def from_env(cls) -> "LocalConfig":
        backend = os.environ.get("TINKER_LOCAL_BACKEND", "transformers")
        if backend == "unsloth":
            print("[warn] TINKER_LOCAL_BACKEND=unsloth is unsupported; using transformers.", flush=True)
            backend = "transformers"

        import torch

        device_env = os.environ.get("TINKER_LOCAL_DEVICE")
        if device_env is None:
            device_env = "cuda" if torch.cuda.is_available() else "cpu"

        seed_env = os.environ.get("TINKER_LOCAL_SEED")
        if seed_env is None:
            seed = 0
        else:
            seed = int(seed_env)

        ckpt_env = os.environ.get("TINKER_LOCAL_CHECKPOINT_DIR")
        if ckpt_env is None:
            ckpt_env = ".tinker-local/checkpoints"

        max_seq_env = os.environ.get("TINKER_LOCAL_MAX_SEQ_LENGTH")
        if max_seq_env is None:
            max_seq = 4096
        else:
            max_seq = int(max_seq_env)

        load_in_4bit_env = os.environ.get("TINKER_LOCAL_LOAD_IN_4BIT")
        if load_in_4bit_env is None:
            load_in_4bit = True
        else:
            load_in_4bit = load_in_4bit_env == "1"

        return cls(
            backend=backend,
            device=device_env,
            seed=seed,
            checkpoint_dir=ckpt_env,
            max_seq_length=max_seq,
            load_in_4bit=load_in_4bit,
        )
