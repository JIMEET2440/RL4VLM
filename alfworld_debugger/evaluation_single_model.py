"""Model-driven ALFWorld debugger via OpenAI-compatible local endpoint.

This script plugs into the existing ALFWorld debugger architecture
(env wrapper + renderer + trajectory logger) and asks a local endpoint
for the next action at each step.

Example:
    python evaluation_single_model.py \
      --model google/gemma-3-27b-it \
      --base-url http://localhost:8001/v1 \
      --config ../VLM_PPO_ALF/alf-config.yaml \
      --num-episodes 3 \
      --max-steps 50 \
      --log-json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from openai import OpenAI
from PIL import Image

from env_wrapper import AlfWorldEnvWrapper
from logger import TrajectoryLogger
from main import parse_user_action
from renderer import DebugRenderer


def _resolve_config_path(config: str) -> Path:
    candidate = Path(config)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / config
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    script_dir = Path(__file__).resolve().parent
    script_candidate = script_dir / config
    if script_candidate.exists():
        return script_candidate.resolve()

    parent_candidate = script_dir.parent / config
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
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


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


def _build_alf_prompt(state, action_only: bool = False) -> str:
    admissible = "\n ".join(f"'{cmd}'" for cmd in state.admissible_actions)
    prompt = "You are an expert in the ALFRED embodied environment. "
    prompt += f"Task: {state.goal}. "
    prompt += f"Current scene text: {state.text_observation}. "
    prompt += f"Admissible actions: [{admissible}]. "
    prompt += "Return valid JSON only. "
    if action_only:
        prompt += 'Format: {"action": "<one admissible action>"}'
    else:
        prompt += (
            'Format: {"thoughts": "<brief reasoning>", '
            '"action": "<one admissible action>"}'
        )
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
    if output is None:
        return ""
    return str(output).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALFWorld debugger using a model endpoint for action proposals"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name exposed by endpoint")
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="OpenAI-compatible base URL (for example http://localhost:8001/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for endpoint auth if required",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../VLM_PPO_ALF/alf-config.yaml",
        help="Path to ALFWorld config YAML",
    )
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    metrics_path = output_dir / f"{model_short}_alf_debug_{timestamp}.json"
    trajectory_path = output_dir / f"{model_short}_trajectory_{timestamp}.json"

    config_path = _resolve_config_path(args.config)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    renderer = DebugRenderer(
        image_dir=str(output_dir / "images"), save_images=not args.no_save_images
    )
    logger = TrajectoryLogger(enabled=args.log_json, output_path=str(trajectory_path))
    env = AlfWorldEnvWrapper(config_path=str(config_path), train_eval="eval", batch_size=1)

    print("=" * 72)
    print("ALFWorld model-driven debugger")
    print(f"Model      : {args.model}")
    print(f"Base URL   : {args.base_url}")
    print(f"Config     : {config_path}")
    print(f"Episodes   : {args.num_episodes}")
    print(f"Max steps  : {args.max_steps}")
    print(f"Vision I/O : {not args.text_only}")
    print("=" * 72)

    episode_rewards = []
    episode_success = []
    episode_lengths = []
    episode_final_goal_condition = []
    episode_max_goal_condition = []
    inference_latencies = []
    total_steps = 0
    image_available_steps = 0
    image_sent_steps = 0
    run_start = time.time()

    try:
        for ep_idx in range(args.num_episodes):
            state, raw_reset = env.reset(seed=args.seed + ep_idx)
            print("#"*80)
            print(state)
            print("#"*80)
            print(raw_reset)
            print("#"*80)

            state.image_path = renderer.save_image(state.rgb_image, state.step_id)
            running_reward = 0.0
            ep_steps = 0
            ep_final_won = 0.0
            ep_final_goal_condition = 0.0
            ep_max_goal_condition = 0.0

            print(f"\n[Episode {ep_idx + 1}/{args.num_episodes}]")
            renderer.print_structure("RESET OUTPUT STRUCTURE", raw_reset)

            for _ in range(args.max_steps):
                renderer.print_state(state)
                renderer.print_actions(state.admissible_actions)

                if not state.admissible_actions:
                    renderer.print_error("No admissible actions returned by environment.")
                    break

                prompt_text = _build_alf_prompt(state, action_only=args.action_only_prompt)
                image_available = state.rgb_image is not None
                image_sent = (not args.text_only) and image_available
                if image_available:
                    image_available_steps += 1
                if image_sent:
                    image_sent_steps += 1

                t0 = time.time()
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
                latency = time.time() - t0
                inference_latencies.append(latency)

                print("\nMODEL OUTPUT")
                print(raw_output)

                action_candidate = _extract_action_candidate(raw_output)
                parse_result = parse_user_action(action_candidate, state.admissible_actions)
                debug_payload = {
                    "model_raw_output": raw_output,
                    "action_candidate": action_candidate,
                    "endpoint_latency_sec": latency,
                    "image_available": image_available,
                    "image_sent": image_sent,
                    "parser_debug": parse_result.debug,
                }
                renderer.print_action_parse_debug(debug_payload)

                if not parse_result.valid or parse_result.action is None:
                    chosen_action = state.admissible_actions[0]
                    print(
                        "[WARNING] Invalid model action; fallback to first admissible action: "
                        f"{chosen_action}"
                    )
                else:
                    chosen_action = parse_result.action

                prev_state = state
                step_result = env.step(chosen_action)
                next_state = step_result.next_state
                next_state.image_path = renderer.save_image(next_state.rgb_image, next_state.step_id)

                renderer.print_structure("STEP OUTPUT STRUCTURE", step_result.raw_step_output)
                renderer.print_structure("STEP INFO STRUCTURE", step_result.info)
                renderer.print_transition(
                    chosen_action, step_result.reward, step_result.done, step_result.info
                )

                logger.log_step(
                    step_id=next_state.step_id,
                    observation_text=prev_state.text_observation,
                    action=chosen_action,
                    reward=step_result.reward,
                    done=step_result.done,
                    info={
                        "env_info": step_result.info,
                        "model_debug": debug_payload,
                    },
                    image_path=next_state.image_path,
                    parse_debug=parse_result.debug,
                )

                won_raw = step_result.info.get("won", 0.0)
                gc_raw = step_result.info.get("goal_condition_success_rate", 0.0)
                try:
                    ep_final_won = float(won_raw[0] if isinstance(won_raw, list) and won_raw else won_raw)
                except Exception:
                    ep_final_won = 0.0
                try:
                    ep_final_goal_condition = float(
                        gc_raw[0] if isinstance(gc_raw, list) and gc_raw else gc_raw
                    )
                except Exception:
                    ep_final_goal_condition = 0.0
                ep_max_goal_condition = max(ep_max_goal_condition, ep_final_goal_condition)

                running_reward += float(step_result.reward)
                ep_steps += 1
                total_steps += 1
                state = next_state

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

                if step_result.done:
                    break

                print("#"*80)
                print(state)
                print("#"*80)
            episode_rewards.append(running_reward)
            episode_success.append(ep_final_won)
            episode_lengths.append(ep_steps)
            episode_final_goal_condition.append(ep_final_goal_condition)
            episode_max_goal_condition.append(ep_max_goal_condition)

            print(
                f"Episode summary -> reward: {running_reward:.2f}, "
                f"success(won): {episode_success[-1]:.0f}, "
                f"goal_condition(final/max): {ep_final_goal_condition:.3f}/{ep_max_goal_condition:.3f}, "
                f"steps: {ep_steps}"
            )

    finally:
        logger.flush()
        env.close()

    elapsed = time.time() - run_start
    metrics = {
        "model": args.model,
        "base_url": args.base_url,
        "config": str(config_path),
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "total_steps": total_steps,
        "total_time_seconds": elapsed,
        "fps": (total_steps / elapsed) if elapsed > 0 else 0.0,
        "episode_reward": {
            "mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "median": float(np.median(episode_rewards)) if episode_rewards else 0.0,
            "min": float(np.min(episode_rewards)) if episode_rewards else 0.0,
            "max": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        },
        "episode_success_rate": float(np.mean(episode_success)) if episode_success else 0.0,
        "goal_condition_success_rate": {
            "final_mean": float(np.mean(episode_final_goal_condition)) if episode_final_goal_condition else 0.0,
            "final_median": float(np.median(episode_final_goal_condition)) if episode_final_goal_condition else 0.0,
            "max_mean": float(np.mean(episode_max_goal_condition)) if episode_max_goal_condition else 0.0,
            "max_median": float(np.median(episode_max_goal_condition)) if episode_max_goal_condition else 0.0,
        },
        "episode_length": {
            "mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "median": float(np.median(episode_lengths)) if episode_lengths else 0.0,
            "min": int(np.min(episode_lengths)) if episode_lengths else 0,
            "max": int(np.max(episode_lengths)) if episode_lengths else 0,
        },
        "inference_latency_sec": {
            "mean": float(np.mean(inference_latencies)) if inference_latencies else 0.0,
            "median": float(np.median(inference_latencies)) if inference_latencies else 0.0,
            "p95": float(np.percentile(inference_latencies, 95)) if inference_latencies else 0.0,
        },
        "image_input": {
            "text_only": bool(args.text_only),
            "steps_with_image_available": int(image_available_steps),
            "steps_with_image_sent": int(image_sent_steps),
            "sent_rate_when_vision_enabled": (
                float(image_sent_steps / total_steps)
                if (total_steps > 0 and not args.text_only)
                else 0.0
            ),
        },
        "trajectory_log_path": str(trajectory_path) if args.log_json else None,
        "timestamp": datetime.now().isoformat(),
    }

    with metrics_path.open("w", encoding="utf-8") as writer:
        json.dump(metrics, writer, indent=2, ensure_ascii=True)

    print("\n" + "=" * 72)
    print("Run summary")
    print(f"Metrics path: {metrics_path}")
    if args.log_json:
        print(f"Trajectory path: {trajectory_path}")
    print(f"Reward mean: {metrics['episode_reward']['mean']:.3f}")
    print(f"Success rate: {metrics['episode_success_rate'] * 100:.2f}%")
    print(
        "Goal-condition success (final/max mean): "
        f"{metrics['goal_condition_success_rate']['final_mean']:.3f}/"
        f"{metrics['goal_condition_success_rate']['max_mean']:.3f}"
    )
    print(
        "Image input (available/sent): "
        f"{metrics['image_input']['steps_with_image_available']}/"
        f"{metrics['image_input']['steps_with_image_sent']}"
    )
    print(f"Latency mean: {metrics['inference_latency_sec']['mean']:.3f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()

