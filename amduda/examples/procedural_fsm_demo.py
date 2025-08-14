"""Demonstration of procedural FSM token streaming.

This sample mirrors the Rust `ProceduralFsm` by stepping through a small
sequence of runtime events.  The backend is selected from the
``AUREX_BACKEND`` environment variable and falls back to ``cpu`` when the
requested accelerator is unavailable.
"""

from enum import Enum
import os


class State(Enum):
    """Enumeration of FSM states."""

    FETCH_TOKEN = "FetchToken"
    KVCACHE_UPDATE = "KVCacheUpdate"
    COMPUTE_ATTENTION = "ComputeAttention"
    OUTPUT_TOKEN = "OutputToken"
    ERROR = "Error"


class ProceduralFsm:
    """Tiny Python port of the Rust procedural FSM."""

    def __init__(self) -> None:
        self.state = State.FETCH_TOKEN

    def on_event(self, event: tuple) -> State:
        name, value = event
        if self.state is State.FETCH_TOKEN and name == "token_fetched":
            self.state = State.COMPUTE_ATTENTION if value else State.KVCACHE_UPDATE
        elif self.state is State.KVCACHE_UPDATE and name == "cache_updated":
            self.state = State.COMPUTE_ATTENTION
        elif self.state is State.COMPUTE_ATTENTION and name == "attention_computed":
            self.state = State.OUTPUT_TOKEN
        elif self.state is State.OUTPUT_TOKEN and name == "token_emitted":
            self.state = State.FETCH_TOKEN
        elif name == "error":
            self.state = State.ERROR
        return self.state


def select_backend() -> str:
    """Choose a backend and default to CPU when unavailable."""

    requested = os.environ.get("AUREX_BACKEND", "").lower()
    available = {"rocm", "vulkan", "opencl", "sycl"}
    return requested if requested in available else "cpu"


def main() -> None:
    backend = select_backend()
    print(f"Running procedural FSM demo on {backend.upper()} backend")
    fsm = ProceduralFsm()
    sequence = [
        ("token_fetched", True),
        ("attention_computed", None),
        ("token_emitted", None),
        ("token_fetched", False),
        ("cache_updated", None),
        ("attention_computed", None),
    ]
    for ev in sequence:
        state = fsm.on_event(ev)
        print(f"after {ev[0]} -> {state.value}")


if __name__ == "__main__":  # pragma: no cover - demo script
    main()
