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


def _build_task_config(base_config_path: Path, task_types: list[int], tag: str, max_steps: int) -> Path:
    with base_config_path.open("r", encoding="utf-8") as reader:
        config = yaml.safe_load(reader)

    env_cfg = config.setdefault("env", {})
    env_cfg["task_types"] = task_types

    # Keep ALF's internal episode cap aligned with evaluator --max-steps.
    # Without this, base config values (often 50) silently truncate episodes.
    rl_cfg = config.setdefault("rl", {})
    rl_train = rl_cfg.setdefault("training", {})
    rl_train["max_nb_steps_per_episode"] = int(max_steps)

    dagger_cfg = config.setdefault("dagger", {})
    dagger_train = dagger_cfg.setdefault("training", {})
    dagger_train["max_nb_steps_per_episode"] = int(max_steps)

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


def _task_family_from_gamefile(info: Dict[str, Any]) -> str:
    gamefile = info.get("extra.gamefile", "")
    if isinstance(gamefile, list) and gamefile:
        gamefile = gamefile[0]
    gamefile = str(gamefile).lower()

    if "look_at_obj_in_light" in gamefile:
        return "look"
    if "pick_and_place" in gamefile:
        return "pick"
    if "pick_two_obj_and_place" in gamefile:
        return "pick_two"
    if "pick_clean_then_place_in_recep" in gamefile:
        return "clean"
    if "pick_heat_then_place_in_recep" in gamefile:
        return "heat"
    if "pick_cool_then_place_in_recep" in gamefile:
        return "cool"
    return "unknown"


def run_task_eval(
    task_name: str,
    task_types: list[int],
    episodes: int,
    seed_base: int,
    max_steps: int,
    client: OpenAI,
    args: argparse.Namespace,
    image_root: Path,
    episode_json_root: Optional[Path] = None,
    progress_path: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg_path = _build_task_config(_resolve_path(args.config), task_types, task_name, max_steps)
    env = AlfWorldEnvWrapper(config_path=str(cfg_path), train_eval="eval", batch_size=1)

    success_count = 0
    episode_lengths = []

    try:
        for ep_idx in range(episodes):
            # Keep resetting until we get the exact requested task family.
            # This guards against any upstream split/task sampler behavior.
            matched = False
            reset_attempts = 0
            while not matched and reset_attempts < 200:
                state, raw_reset = env.reset(seed=seed_base + ep_idx + reset_attempts)
                _, reset_info = raw_reset
                family = _task_family_from_gamefile(reset_info)
                matched = family == task_name
                reset_attempts += 1

            if not matched:
                raise RuntimeError(
                    f"Could not sample '{task_name}' task after {reset_attempts} resets. "
                    f"Check task_types in generated config: {cfg_path}"
                )

            ep_dir = image_root / task_name / f"episode_{ep_idx + 1:04d}"
            if not args.no_save_images:
                _save_image(state.rgb_image, ep_dir / "step_0000.png")

            current_state_payload = {
                "step_id": int(state.step_id),
                "goal": state.goal,
                "text_observation": state.text_observation,
                "inventory": state.inventory,
                "admissible_actions": state.admissible_actions,
                "image_path": str(ep_dir / "step_0000.png") if not args.no_save_images else None,
                "image_available": bool(state.rgb_image is not None),
            }

            ep_success = False
            ep_steps = 0
            step_traces = []

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

                next_state_payload = {
                    "step_id": int(state.step_id),
                    "goal": state.goal,
                    "text_observation": state.text_observation,
                    "inventory": state.inventory,
                    "admissible_actions": state.admissible_actions,
                    "image_path": str(ep_dir / f"step_{step_idx:04d}.png") if not args.no_save_images else None,
                    "image_available": bool(state.rgb_image is not None),
                }

                step_traces.append(
                    {
                        "step_index": int(step_idx),
                        "current_state": current_state_payload,
                        "model_output_text": raw_output,
                        "parsed_action_candidate": action_candidate,
                        "action_parse_valid": bool(parse_result.valid),
                        "action_parse_debug": parse_result.debug,
                        "chosen_action": chosen_action,
                        "transition": {
                            "goal_condition_progress": _to_float_first(
                                step_result.info.get("goal_condition_success_rate", 0.0)
                            ),
                            "done": bool(step_result.done),
                            "won": _to_float_first(step_result.info.get("won", 0.0)),
                            "goal_condition_success_rate": _to_float_first(
                                step_result.info.get("goal_condition_success_rate", 0.0)
                            ),
                            "info": step_result.info,
                        },
                        "next_state": next_state_payload,
                    }
                )
                current_state_payload = next_state_payload

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

            episode_payload = {
                "task_name": task_name,
                "episode_index": int(ep_idx + 1),
                "seed": int(seed_base + ep_idx),
                "success": bool(ep_success),
                "steps_taken": int(ep_steps),
                "steps": step_traces,
            }

            latest_episode_json = None
            if episode_json_root is not None:
                task_episode_dir = episode_json_root / task_name
                task_episode_dir.mkdir(parents=True, exist_ok=True)
                episode_out_path = task_episode_dir / f"episode_{ep_idx + 1:04d}.json"
                with episode_out_path.open("w", encoding="utf-8") as writer:
                    json.dump(episode_payload, writer, indent=2, ensure_ascii=True)
                latest_episode_json = str(episode_out_path)

            print(
                f"[{task_name}] episode {ep_idx + 1}/{episodes} -> "
                f"success={int(ep_success)} steps={ep_steps}",
                flush=True,
            )

            if progress_path is not None:
                progress_payload = {
                    "task_name": task_name,
                    "task_types": task_types,
                    "episodes_target": int(episodes),
                    "episodes_completed": int(ep_idx + 1),
                    "success_count": int(success_count),
                    "success_rate_so_far": (success_count / (ep_idx + 1)) if (ep_idx + 1) > 0 else 0.0,
                    "episode_length_mean_so_far": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                    "episode_length_median_so_far": float(np.median(episode_lengths)) if episode_lengths else 0.0,
                    "latest_episode_json": latest_episode_json,
                    "timestamp": datetime.now().isoformat(),
                }
                with progress_path.open("w", encoding="utf-8") as writer:
                    json.dump(progress_payload, writer, indent=2, ensure_ascii=True)
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
    parser = argparse.ArgumentParser(description="Gemma ALFWorld evaluation for all task types")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="EMPTY")

    # Keep defaults aligned with requested configuration.
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
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
    parser.add_argument("--clean-episodes", type=int, default=200)
    parser.add_argument("--heat-episodes", type=int, default=200)
    parser.add_argument("--cool-episodes", type=int, default=200)
    parser.add_argument("--pick-two-episodes", type=int, default=200)
    parser.add_argument("--run-tag", type=str, default=None)
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
    run_tag = args.run_tag if args.run_tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_json_root = output_dir / f"gemma_eval_episode_json_{run_tag}"
    episode_json_root.mkdir(parents=True, exist_ok=True)

    task_runs = [
        ("pick", [1], args.pick_episodes, args.seed + 0),
        ("look", [2], args.look_episodes, args.seed + 100000),
        ("clean", [3], args.clean_episodes, args.seed + 200000),
        ("heat", [4], args.heat_episodes, args.seed + 300000),
        ("cool", [5], args.cool_episodes, args.seed + 400000),
        ("pick_two", [6], args.pick_two_episodes, args.seed + 500000),
    ]

    print("=" * 72, flush=True)
    print("Gemma ALFWorld evaluation (All task types)", flush=True)
    print(f"Model       : {args.model}", flush=True)
    print(f"Base URL    : {args.base_url}", flush=True)
    print(f"Config      : {_resolve_path(args.config)}", flush=True)
    print(f"Pick eps    : {args.pick_episodes}", flush=True)
    print(f"Look eps    : {args.look_episodes}", flush=True)
    print(f"Clean eps   : {args.clean_episodes}", flush=True)
    print(f"Heat eps    : {args.heat_episodes}", flush=True)
    print(f"Cool eps    : {args.cool_episodes}", flush=True)
    print(f"PickTwo eps : {args.pick_two_episodes}", flush=True)
    print(f"Max steps   : {args.max_steps}", flush=True)
    print(f"Text only   : {args.text_only}", flush=True)
    print(f"Image dir   : {image_root}", flush=True)
    print(f"Run tag     : {run_tag}", flush=True)
    print(f"Episode JSON: {episode_json_root}", flush=True)
    print("=" * 72, flush=True)

    metrics_by_task: Dict[str, Dict[str, Any]] = {}
    for task_name, task_types, episodes, seed_base in task_runs:
        if episodes <= 0:
            metrics_by_task[task_name] = {
                "task_name": task_name,
                "task_types": task_types,
                "episodes": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "episode_length_mean": 0.0,
                "episode_length_median": 0.0,
            }
            print(f"[{task_name}] skipped (episodes=0)", flush=True)
            continue

        metrics_by_task[task_name] = run_task_eval(
            task_name=task_name,
            task_types=task_types,
            episodes=episodes,
            seed_base=seed_base,
            max_steps=args.max_steps,
            client=client,
            args=args,
            image_root=image_root,
            episode_json_root=episode_json_root,
            progress_path=output_dir / f"gemma_eval_{task_name}_progress_{run_tag}.json",
        )

    finished = time.time()

    total_success = int(sum(task_metrics["success_count"] for task_metrics in metrics_by_task.values()))
    total_episodes = int(sum(task_metrics["episodes"] for task_metrics in metrics_by_task.values()))

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
        "episode_json_root": str(episode_json_root),
        "tasks": metrics_by_task,
        "total_success_count": total_success,
        "total_episodes": total_episodes,
        "runtime_seconds": finished - started,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = output_dir / f"gemma_eval_all_tasks_{run_tag}.json"
    with out_path.open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2, ensure_ascii=True)

    print("\n" + "=" * 72, flush=True)
    print("Evaluation complete", flush=True)
    for task_name in ["pick", "look", "clean", "heat", "cool", "pick_two"]:
        task_metrics = metrics_by_task.get(task_name, {})
        print(
            f"{task_name} success: {task_metrics.get('success_count', 0)}/{task_metrics.get('episodes', 0)}",
            flush=True,
        )
    print(f"Total success: {total_success}/{total_episodes}", flush=True)
    print(f"Summary JSON: {out_path}", flush=True)
    print(f"Per-episode JSON root: {episode_json_root}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
