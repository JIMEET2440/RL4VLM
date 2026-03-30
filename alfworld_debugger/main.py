import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from env_wrapper import AlfWorldEnvWrapper
from logger import TrajectoryLogger
from renderer import DebugRenderer


@dataclass
class ActionParseResult:
    valid: bool
    should_quit: bool
    action: Optional[str]
    debug: Dict[str, Any]
    error: Optional[str] = None


def parse_user_action(user_input: str, admissible_actions: List[str]) -> ActionParseResult:
    raw = user_input
    text = user_input.strip()
    debug: Dict[str, Any] = {
        "raw_input": raw,
        "trimmed_input": text,
        "num_admissible_actions": len(admissible_actions),
    }

    if text.lower() in {"q", "quit", "exit"}:
        debug["decision"] = "quit"
        return ActionParseResult(valid=False, should_quit=True, action=None, debug=debug)

    if re.fullmatch(r"\d+", text):
        idx = int(text)
        debug["parsing_mode"] = "index"
        debug["parsed_index"] = idx
        if idx < 0 or idx >= len(admissible_actions):
            return ActionParseResult(
                valid=False,
                should_quit=False,
                action=None,
                debug=debug,
                error=f"Index {idx} out of range [0, {len(admissible_actions) - 1}]",
            )
        selected = admissible_actions[idx]
        debug["selected_action"] = selected
        return ActionParseResult(valid=True, should_quit=False, action=selected, debug=debug)

    debug["parsing_mode"] = "text"
    candidate = _extract_json_action(text) or text
    debug["candidate_text"] = candidate
    normalized_candidate = _normalize(candidate)

    normalized_map = {_normalize(action): action for action in admissible_actions}
    if normalized_candidate in normalized_map:
        selected = normalized_map[normalized_candidate]
        debug["match_type"] = "exact_normalized"
        debug["selected_action"] = selected
        return ActionParseResult(valid=True, should_quit=False, action=selected, debug=debug)

    contains_matches = [
        action
        for action in admissible_actions
        if normalized_candidate and normalized_candidate in _normalize(action)
    ]
    debug["contains_matches"] = contains_matches
    if len(contains_matches) == 1:
        selected = contains_matches[0]
        debug["match_type"] = "unique_partial"
        debug["selected_action"] = selected
        return ActionParseResult(valid=True, should_quit=False, action=selected, debug=debug)

    if len(contains_matches) > 1:
        return ActionParseResult(
            valid=False,
            should_quit=False,
            action=None,
            debug=debug,
            error=(
                "Ambiguous text action. Multiple admissible actions matched your input. "
                "Use an index or a more specific action string."
            ),
        )

    return ActionParseResult(
        valid=False,
        should_quit=False,
        action=None,
        debug=debug,
        error="Input did not match any admissible action.",
    )


def _extract_json_action(text: str) -> Optional[str]:
    pattern = r'"action"\s*:\s*"([^"]+)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _default_debug_outputs_dir() -> Path:
    return Path(__file__).resolve().parent / "debug_outputs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive ALFWorld debugging tool")
    parser.add_argument(
        "--config",
        type=str,
        default="VLM_PPO_ALF/alf-config.yaml",
        help="Path to ALFWorld config YAML.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Environment seed for reset().")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Safety cap on steps per interactive episode.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=str(_default_debug_outputs_dir() / "images"),
        help="Directory to save step images.",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Disable saving RGB image snapshots each step.",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="If set, write a trajectory JSON log.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(_default_debug_outputs_dir() / "trajectory.json"),
        help="JSON trajectory output path.",
    )
    return parser


def run() -> None:
    args = build_arg_parser().parse_args()

    config_path = _resolve_config_path(args.config)

    renderer = DebugRenderer(
        image_dir=args.image_dir,
        save_images=not args.no_save_images,
    )
    logger = TrajectoryLogger(enabled=args.log_json, output_path=args.log_path)

    env = AlfWorldEnvWrapper(config_path=str(config_path), train_eval="eval", batch_size=1)

    try:
        current_state, raw_reset_output = env.reset(seed=args.seed)
        current_state.image_path = renderer.save_image(current_state.rgb_image, current_state.step_id)

        print("\nALFWorld Interactive Debugger")
        print(f"Session start: {datetime.now().isoformat(timespec='seconds')}")
        print("Type an action index, action text, or 'quit' to exit.")

        renderer.print_structure("RESET OUTPUT STRUCTURE", raw_reset_output)

        total_steps = 0
        while total_steps < args.max_steps:
            renderer.print_state(current_state)
            renderer.print_actions(current_state.admissible_actions)

            if not current_state.admissible_actions:
                renderer.print_error("No admissible actions provided by env; stopping.")
                break

            user_input = input("Enter action index/text (or 'quit'): ")
            parse_result = parse_user_action(user_input, current_state.admissible_actions)
            renderer.print_action_parse_debug(parse_result.debug)

            if parse_result.should_quit:
                print("Manual quit requested.")
                break

            if not parse_result.valid or parse_result.action is None:
                renderer.print_error(parse_result.error or "Invalid action input.")
                continue

            action = parse_result.action
            o_t = current_state
            step_result = env.step(action)
            next_state = step_result.next_state
            next_state.image_path = renderer.save_image(next_state.rgb_image, next_state.step_id)

            renderer.print_structure("STEP OUTPUT STRUCTURE", step_result.raw_step_output)
            renderer.print_transition(action, step_result.reward, step_result.done, step_result.info)

            logger.log_step(
                step_id=next_state.step_id,
                observation_text=o_t.text_observation,
                action=action,
                reward=step_result.reward,
                done=step_result.done,
                info=step_result.info,
                image_path=next_state.image_path,
                parse_debug=parse_result.debug,
            )

            # Explicitly track RL tuple variables for learning/debugging.
            print("\nTracked Variables")
            print(f"o_t text: {o_t.text_observation[:240]}")
            print(f"a_t: {action}")
            print(f"r_t: {step_result.reward}")
            print(f"o_t+1 text: {next_state.text_observation[:240]}")
            print(f"done: {step_result.done}")
            print(f"info keys: {list(step_result.info.keys())}")

            total_steps += 1
            current_state = next_state

            if step_result.done:
                print("Episode finished (done=True).")
                break

        if total_steps >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}; stopping session.")

    finally:
        logger.flush()
        renderer.close()
        env.close()
        if args.log_json:
            print(f"Trajectory log saved: {Path(args.log_path).resolve()}")


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

    # Let wrapper raise a clear file-not-found message.
    return candidate


if __name__ == "__main__":
    run()
