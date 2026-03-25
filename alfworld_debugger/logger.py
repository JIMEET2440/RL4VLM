import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Handle numpy / torch scalar-like values without importing those packages.
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            return item_fn()
        except Exception:
            pass

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            return {
                "type": type(value).__name__,
                "shape": list(shape),
                "dtype": str(dtype),
            }
        except Exception:
            pass

    return repr(value)


class TrajectoryLogger:
    def __init__(self, enabled: bool, output_path: str):
        self.enabled = enabled
        self.output_path = Path(output_path)
        self.records: List[Dict[str, Any]] = []

        if self.enabled:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(
        self,
        step_id: int,
        observation_text: str,
        action: str,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        image_path: Optional[str],
        parse_debug: Dict[str, Any],
    ) -> None:
        if not self.enabled:
            return

        self.records.append(
            {
                "step_id": step_id,
                "observation_text": observation_text,
                "action": action,
                "reward": reward,
                "done": done,
                "info": _to_jsonable(info),
                "image_path": image_path,
                "action_parse_debug": _to_jsonable(parse_debug),
            }
        )

    def flush(self) -> None:
        if not self.enabled:
            return

        payload = {
            "num_steps": len(self.records),
            "trajectory": self.records,
        }
        with self.output_path.open("w", encoding="utf-8") as writer:
            json.dump(payload, writer, indent=2, ensure_ascii=True)
