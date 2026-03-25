from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _HAS_RICH = True
except Exception:
    _HAS_RICH = False
    Console = None
    Panel = None
    Table = None

try:
    from PIL import Image

    _HAS_PIL = True
except Exception:
    _HAS_PIL = False
    Image = None


def summarize_structure(value: Any, depth: int = 0, max_depth: int = 3) -> Any:
    if depth >= max_depth:
        return f"<{type(value).__name__}>"

    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        # Works for torch tensors without importing torch.
        try:
            shape = list(value.shape)
            dtype = str(value.dtype)
            return {"type": type(value).__name__, "shape": shape, "dtype": dtype}
        except Exception:
            pass

    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys()),
            "items": {
                k: summarize_structure(v, depth=depth + 1, max_depth=max_depth)
                for k, v in value.items()
            },
        }

    if isinstance(value, (list, tuple)):
        preview = [summarize_structure(v, depth=depth + 1, max_depth=max_depth) for v in value[:3]]
        return {
            "type": type(value).__name__,
            "len": len(value),
            "preview": preview,
        }

    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"type": type(value).__name__, "value": value}

    return {"type": type(value).__name__, "repr": repr(value)[:160]}


class DebugRenderer:
    def __init__(self, image_dir: str, save_images: bool = True):
        self.image_dir = Path(image_dir)
        self.save_images = save_images
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console() if _HAS_RICH else None

    def save_image(self, rgb_image: Optional[np.ndarray], step_id: int) -> Optional[str]:
        if not self.save_images or rgb_image is None:
            return None
        if not _HAS_PIL:
            return None

        path = self.image_dir / f"step_{step_id:04d}.png"
        image_array = np.asarray(rgb_image)
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        Image.fromarray(image_array).save(path)
        return str(path)

    def print_state(self, state: Any) -> None:
        step_header = f"STEP: {state.step_id}"
        if _HAS_RICH:
            self.console.rule(f"[bold cyan]{step_header}")
        else:
            print("\n" + "=" * 72)
            print(step_header)
            print("=" * 72)

        body_lines = [
            f"TASK: {state.goal}",
            "",
            "TEXT OBSERVATION:",
            state.text_observation,
            "",
            f"INVENTORY: {state.inventory}",
        ]

        if state.image_path:
            body_lines.append(f"RGB IMAGE SAVED: {state.image_path}")
            if state.rgb_image is not None:
                body_lines.append(f"IMAGE SHAPE: {list(state.rgb_image.shape)}")

        text = "\n".join(body_lines)
        if _HAS_RICH:
            self.console.print(Panel(text, title="Current State", expand=False))
        else:
            print(text)

    def print_actions(self, actions: List[str]) -> None:
        if _HAS_RICH:
            table = Table(title="AVAILABLE ACTIONS", show_lines=False)
            table.add_column("Index", justify="right", style="cyan")
            table.add_column("Action", style="white")
            for idx, action in enumerate(actions):
                table.add_row(str(idx), action)
            self.console.print(table)
        else:
            print("AVAILABLE ACTIONS:")
            for idx, action in enumerate(actions):
                print(f"[{idx}] {action}")

    def print_structure(self, label: str, value: Any) -> None:
        summary = summarize_structure(value)
        text = json.dumps(summary, indent=2, ensure_ascii=True)
        if _HAS_RICH:
            self.console.print(Panel(text, title=label, expand=False))
        else:
            print(f"\n{label}:\n{text}")

    def print_action_parse_debug(self, parse_debug: Dict[str, Any]) -> None:
        text = json.dumps(parse_debug, indent=2, ensure_ascii=True)
        if _HAS_RICH:
            self.console.print(Panel(text, title="ACTION PARSING DEBUG", expand=False))
        else:
            print("\nACTION PARSING DEBUG:")
            print(text)

    def print_transition(self, action: str, reward: float, done: bool, info: Dict[str, Any]) -> None:
        line = f"[STATE] -> [ACTION: {action}] -> [ENV STEP] -> [NEXT STATE]"
        info_keys = list(info.keys())
        text = (
            f"{line}\n"
            f"reward: {reward}\n"
            f"done: {done}\n"
            f"info keys: {info_keys}"
        )
        if _HAS_RICH:
            self.console.print(Panel(text, title="TRANSITION", expand=False))
        else:
            print("\nTRANSITION:")
            print(text)

    def print_error(self, message: str) -> None:
        if _HAS_RICH:
            self.console.print(f"[bold red]ERROR:[/bold red] {message}")
        else:
            print(f"ERROR: {message}")
