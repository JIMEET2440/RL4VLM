from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from openai import OpenAI
from PIL import Image
import yaml

# Reuse the existing ALF wrapper/action parser from alfworld_debugger.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEBUGGER_DIR = ROOT_DIR / "alfworld_debugger"
if str(DEBUGGER_DIR) not in sys.path:
    sys.path.insert(0, str(DEBUGGER_DIR))

from env_wrapper import AlfWorldEnvWrapper  # noqa: E402
from main import parse_user_action  # noqa: E402


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    script_candidate = SCRIPT_DIR / path
    if script_candidate.exists():
        return script_candidate.resolve()

    parent_candidate = ROOT_DIR / path
    if parent_candidate.exists():
        return parent_candidate.resolve()

    return candidate


def _numpy_rgb_to_b64_png(rgb_image: np.ndarray) -> str:
    array = np.asarray(rgb_image)
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_first_json_blob(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = start
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if depth != 0:
        return None

    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _extract_action_candidate(text: str) -> str:
    parsed = _extract_first_json_blob(text)
    if parsed and "action" in parsed:
        action = str(parsed.get("action", "")).strip()
        if action:
            return action

    match = re.search(r'"action"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1).strip()

    return text.strip()


def _build_prompt(goal: str, observation: str, admissible_actions: list[str], action_only: bool) -> str:
    admissible = "\n ".join(f"'{cmd}'" for cmd in admissible_actions)
    prompt = "You are an expert in the ALFRED embodied environment. "
    prompt += f"Task: {goal}. "
    prompt += f"Current scene text: {observation}. "
    prompt += f"Admissible actions: [{admissible}]. "
    prompt += "Return valid JSON only. "
    if action_only:
        prompt += 'Format: {"action": "<one admissible action>"}'
    else:
        prompt += 'Format: {"thoughts": "<brief reasoning>", "action": "<one admissible action>"}'
    return prompt


def _call_model(
    client: OpenAI,
    model_name: str,
    prompt_text: str,
    rgb_image: Optional[np.ndarray],
    use_image: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    content = [{"type": "text", "text": prompt_text}]

    if use_image and rgb_image is not None:
        image_b64 = _numpy_rgb_to_b64_png(rgb_image)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    if not response.choices or not response.choices[0].message:
        raise ValueError(f"Empty response from endpoint: {response}")

    output = response.choices[0].message.content
    return "" if output is None else str(output).strip()


def _save_image(rgb_image: Optional[np.ndarray], path: Path) -> bool:
    if rgb_image is None:
        return False
    array = np.asarray(rgb_image)
    array = np.clip(array, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return True


def _build_task_config(base_config_path: Path, task_types: list[int], tag: str) -> Path:
    with base_config_path.open("r", encoding="utf-8") as reader:
        config = yaml.safe_load(reader)

    env_cfg = config.setdefault("env", {})
    env_cfg["task_types"] = task_types

    tmp_dir = SCRIPT_DIR / ".tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"alf-config-{tag}.yaml"
    with out_path.open("w", encoding="utf-8") as writer:
        yaml.safe_dump(config, writer, sort_keys=False)
    return out_path


def _to_float_first(value: Any, default: float = 0.0) -> float:
    raw = value[0] if isinstance(value, list) and value else value
    try:
        return float(raw)
    except Exception:
        return default


def run_task_eval(
    task_name: str,
    task_types: list[int],
    episodes: int,
    seed_base: int,
    max_steps: int,
    client: OpenAI,
    args: argparse.Namespace,
    image_root: Path,
) -> Dict[str, Any]:
    cfg_path = _build_task_config(_resolve_path(args.config), task_types, task_name)
    env = AlfWorldEnvWrapper(config_path=str(cfg_path), train_eval="eval", batch_size=1)

    success_count = 0
    episode_lengths = []

    try:
        for ep_idx in range(episodes):
            state, _ = env.reset(seed=seed_base + ep_idx)
            ep_dir = image_root / task_name / f"episode_{ep_idx + 1:04d}"
            if not args.no_save_images:
                _save_image(state.rgb_image, ep_dir / "step_0000.png")

            ep_success = False
            ep_steps = 0

            for step_idx in range(1, max_steps + 1):
                if not state.admissible_actions:
                    break

                prompt_text = _build_prompt(
                    goal=state.goal,
                    observation=state.text_observation,
                    admissible_actions=state.admissible_actions,
                    action_only=args.action_only_prompt,
                )

                raw_output = _call_model(
                    client=client,
                    model_name=args.model,
                    prompt_text=prompt_text,
                    rgb_image=state.rgb_image,
                    use_image=not args.text_only,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                action_candidate = _extract_action_candidate(raw_output)
                parse_result = parse_user_action(action_candidate, state.admissible_actions)
                if parse_result.valid and parse_result.action is not None:
                    chosen_action = parse_result.action
                else:
                    chosen_action = state.admissible_actions[0]

                step_result = env.step(chosen_action)
                state = step_result.next_state
                if not args.no_save_images:
                    _save_image(state.rgb_image, ep_dir / f"step_{step_idx:04d}.png")

                ep_steps = step_idx
                won = _to_float_first(step_result.info.get("won", 0.0))
                if won > 0.5:
                    ep_success = True
                    break
                if step_result.done:
                    break

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            episode_lengths.append(ep_steps)
            if ep_success:
                success_count += 1

            print(
                f"[{task_name}] episode {ep_idx + 1}/{episodes} -> "
                f"success={int(ep_success)} steps={ep_steps}"
            )
    finally:
        env.close()

    return {
        "task_name": task_name,
        "task_types": task_types,
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": (success_count / episodes) if episodes > 0 else 0.0,
        "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "episode_length_median": float(np.median(episode_lengths)) if episode_lengths else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma ALFWorld evaluation for Pick and Look tasks")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="EMPTY")

    # Keep defaults aligned with requested configuration.
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--action-only-prompt", action="store_true")
    parser.add_argument("--text-only", action="store_true", help="Do not send image content")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--log-json", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./debug_outputs/model_debug")

    parser.add_argument("--config", type=str, default="../VLM_PPO_ALF/alf-config.yaml")
    parser.add_argument("--pick-episodes", type=int, default=200)
    parser.add_argument("--look-episodes", type=int, default=200)
    parser.add_argument("--image-dir", type=str, default="./images")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_root = _resolve_path(args.image_dir)
    if args.no_save_images:
        # Respect the flag but keep directory available for manual checks.
        image_root.mkdir(parents=True, exist_ok=True)
    else:
        image_root.mkdir(parents=True, exist_ok=True)

    started = time.time()

    print("=" * 72)
    print("Gemma ALFWorld evaluation (Pick + Look)")
    print(f"Model       : {args.model}")
    print(f"Base URL    : {args.base_url}")
    print(f"Config      : {_resolve_path(args.config)}")
    print(f"Pick eps    : {args.pick_episodes}")
    print(f"Look eps    : {args.look_episodes}")
    print(f"Max steps   : {args.max_steps}")
    print(f"Text only   : {args.text_only}")
    print(f"Image dir   : {image_root}")
    print("=" * 72)

    pick_metrics = run_task_eval(
        task_name="pick",
        task_types=[1],
        episodes=args.pick_episodes,
        seed_base=args.seed,
        max_steps=args.max_steps,
        client=client,
        args=args,
        image_root=image_root,
    )

    look_metrics = run_task_eval(
        task_name="look",
        task_types=[2],
        episodes=args.look_episodes,
        seed_base=args.seed + 100000,
        max_steps=args.max_steps,
        client=client,
        args=args,
        image_root=image_root,
    )

    finished = time.time()

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "config": str(_resolve_path(args.config)),
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "text_only": args.text_only,
        "image_dir": str(image_root),
        "pick": pick_metrics,
        "look": look_metrics,
        "total_success_count": pick_metrics["success_count"] + look_metrics["success_count"],
        "total_episodes": args.pick_episodes + args.look_episodes,
        "runtime_seconds": finished - started,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = output_dir / f"gemma_eval_pick_look_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2, ensure_ascii=True)

    print("\n" + "=" * 72)
    print("Evaluation complete")
    print(f"Pick success: {pick_metrics['success_count']}/{pick_metrics['episodes']}")
    print(f"Look success: {look_metrics['success_count']}/{look_metrics['episodes']}")
    print(f"Total success: {summary['total_success_count']}/{summary['total_episodes']}")
    print(f"Summary JSON: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
