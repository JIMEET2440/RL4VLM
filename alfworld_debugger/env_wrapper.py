import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import alfworld.agents.environment as environment
except Exception as exc:  # pragma: no cover - only triggers in missing env setups
    environment = None
    _ALFWORLD_IMPORT_ERROR = exc
else:
    _ALFWORLD_IMPORT_ERROR = None

try:
    from alfworld.agents.utils.misc import get_templated_task_desc
except Exception:
    get_templated_task_desc = None


def _first_item(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


@dataclass
class EnvState:
    step_id: int
    text_observation: str
    goal: str
    inventory: str
    admissible_actions: List[str]
    rgb_image: Optional[np.ndarray]
    raw_observation: Any
    raw_info: Dict[str, Any]
    image_path: Optional[str] = None


@dataclass
class StepResult:
    action: str
    reward: float
    done: bool
    info: Dict[str, Any]
    next_state: EnvState
    raw_step_output: Tuple[Any, Any, Any, Dict[str, Any]]


class AlfWorldEnvWrapper:
    """Thin wrapper around ALFWorld env with consistent state extraction for debugging."""

    def __init__(self, config_path: str, train_eval: str = "eval", batch_size: int = 1):
        if environment is None:
            raise RuntimeError(
                "Failed to import alfworld. Activate your ALFWorld conda env first. "
                f"Import error: {_ALFWORLD_IMPORT_ERROR}"
            )

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as reader:
            self.config = yaml.safe_load(reader)

        env_type = self.config["env"]["type"]
        env_ctor = self._resolve_env_ctor(env_type)
        self.env = self._build_env_with_split_fallback(
            env_ctor=env_ctor,
            train_eval=train_eval,
            batch_size=batch_size,
        )
        self.step_id = 0

    def _build_env_with_split_fallback(self, env_ctor, train_eval: str, batch_size: int):
        split_candidates = [train_eval]
        if train_eval == "eval":
            split_candidates.append("eval_in_distribution")
        elif train_eval == "eval_in_distribution":
            split_candidates.append("eval")

        last_exc: Optional[Exception] = None
        for split in split_candidates:
            try:
                return env_ctor(self.config, train_eval=split).init_env(batch_size=batch_size)
            except Exception as exc:
                last_exc = exc
                if "Invalid split" not in str(exc):
                    raise

        raise RuntimeError(
            "Failed to initialize ALFWorld environment with supported split names "
            f"{split_candidates}. Last error: {last_exc}"
        )

    def _resolve_env_ctor(self, env_type: str):
        """Resolve env constructor across ALFWorld API variants."""
        direct_ctor = getattr(environment, env_type, None)
        if callable(direct_ctor):
            return direct_ctor

        factory = getattr(environment, "get_environment", None)
        if callable(factory):
            try:
                return factory(env_type)
            except Exception as exc:
                raise AttributeError(
                    f"ALFWorld environment type '{env_type}' was not found via get_environment()."
                ) from exc

        available = sorted(
            name
            for name in dir(environment)
            if name.lower().endswith("env") or "environment" in name.lower()
        )
        raise AttributeError(
            "Unable to resolve ALFWorld environment constructor for "
            f"'{env_type}'. Exposed symbols: {available}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[EnvState, Tuple[Any, Dict[str, Any]]]:
        if seed is not None:
            self.env.seed(seed)

        observation, info = self.env.reset()
        self.step_id = 0
        state = self._build_state(observation, info, step_id=self.step_id)
        return state, (observation, info)

    def step(self, action: str) -> StepResult:
        self.step_id += 1
        observation, scores, dones, info = self.env.step([action])
        reward = float(_first_item(scores) if scores is not None else 0.0)
        done = bool(_first_item(dones))

        next_state = self._build_state(observation, info, step_id=self.step_id)
        return StepResult(
            action=action,
            reward=reward,
            done=done,
            info=info,
            next_state=next_state,
            raw_step_output=(observation, scores, dones, info),
        )

    def close(self) -> None:
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()

    def _build_state(self, observation: Any, info: Dict[str, Any], step_id: int) -> EnvState:
        text_obs = _first_item(observation)
        text_obs = text_obs if isinstance(text_obs, str) else _safe_json(text_obs)

        admissible = info.get("admissible_commands", [])
        admissible = _first_item(admissible)
        admissible_actions = admissible if isinstance(admissible, list) else []

        inventory = self._extract_inventory(info)
        goal = self._extract_goal(info)
        rgb_image = self._extract_rgb_image()

        return EnvState(
            step_id=step_id,
            text_observation=text_obs,
            goal=goal,
            inventory=inventory,
            admissible_actions=admissible_actions,
            rgb_image=rgb_image,
            raw_observation=observation,
            raw_info=info,
        )

    def _extract_rgb_image(self) -> Optional[np.ndarray]:
        get_frames = getattr(self.env, "get_frames", None)
        if not callable(get_frames):
            return None

        frames = get_frames()
        if not isinstance(frames, list) or not frames:
            return None

        frame = frames[0]
        if not isinstance(frame, np.ndarray):
            return None

        if frame.ndim == 3 and frame.shape[-1] == 3:
            # ALFWorld THOR frames are BGR in this codebase; convert to RGB.
            return frame[:, :, [2, 1, 0]].copy()
        return frame.copy()

    def _extract_goal(self, info: Dict[str, Any]) -> str:
        for key in ("task", "goal", "goal_desc", "task_desc"):
            if key in info:
                value = _first_item(info[key])
                if isinstance(value, str) and value.strip():
                    return value.strip()

        if get_templated_task_desc is not None:
            try:
                traj_data = self.env.envs[0].traj_data
                return get_templated_task_desc(traj_data)
            except Exception:
                pass

        gamefile = _first_item(info.get("extra.gamefile", ""))
        if isinstance(gamefile, str) and gamefile:
            return f"Goal unavailable; game file: {gamefile}"
        return "Goal unavailable from current info dict."

    def _extract_inventory(self, info: Dict[str, Any]) -> str:
        for key in ("inventory", "inventory_text"):
            if key in info:
                value = _first_item(info[key])
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return "(inventory not provided by env info)"
