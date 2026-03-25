# ALFWorld Interactive Debugger

A terminal-first debugging/learning interface for ALFWorld episodes.

## What It Shows Per Step

- Current text observation (`o_t`)
- Current task goal
- Inventory (if exposed by env info)
- Current RGB frame saved to disk
- Full admissible action list (dynamic action space)
- Transition outputs from `env.step`: reward, done, info keys
- Data flow trace: `[STATE] -> [ACTION] -> [ENV STEP] -> [NEXT STATE]`
- Structure summaries of raw reset/step outputs (types/shapes)
- Optional trajectory JSON logging

## Files

- `main.py`: interactive loop
- `env_wrapper.py`: ALFWorld reset/step adapter and state extraction
- `renderer.py`: terminal rendering and structure summaries
- `logger.py`: trajectory JSON logger

## Run

From `RL4VLM_conda`:

```bash
conda activate vrenv-alf
python alfworld_debugger/main.py \
  --config VLM_PPO_ALF/alf-config.yaml \
  --seed 42 \
  --max-steps 200 \
  --log-json
```

Optional flags:

- `--no-save-images`: do not save RGB frame images
- `--image-dir ./debug_outputs/images`: choose image output directory
- `--log-path ./debug_outputs/trajectory.json`: choose trajectory log path

## Example Interaction

```text
ALFWorld Interactive Debugger
Session start: 2026-03-25T10:15:01
Type an action index, action text, or 'quit' to exit.

STEP: 0
TASK: put a mug in cabinet
TEXT OBSERVATION:
You are in the kitchen. You see a mug on the table...
INVENTORY: (inventory not provided by env info)
RGB IMAGE SAVED: ./debug_outputs/images/step_0000.png

AVAILABLE ACTIONS:
[0] go to cabinet 1
[1] take mug from table 1
[2] open cabinet 1
...

Enter action index/text (or 'quit'): 1

TRANSITION:
[STATE] -> [ACTION: take mug from table 1] -> [ENV STEP] -> [NEXT STATE]
reward: 0.0
done: False
info keys: ['won', 'goal_condition_success_rate', 'admissible_commands', ...]

Tracked Variables
o_t text: You are in the kitchen...
a_t: take mug from table 1
r_t: 0.0
o_t+1 text: You are holding the mug...
done: False
```

## Notes

- This tool is intentionally not an RL trainer; it is for manual stepping and debugging.
- ALFWorld and dependencies must be installed in the active conda environment.
- If `rich` is installed, output is formatted with rich panels/tables; otherwise plain text is used.
