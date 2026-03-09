#!/usr/bin/env python3
"""
eval_sft.py – Evaluate the SFT-finetuned LLaVA-v1.6-Mistral-7B on ALFWorld data.

Usage:
    # Evaluate 200 random samples on GPU 4
    python eval_sft.py --num-samples 200 --gpu 4

    # Evaluate all samples (slow)
    python eval_sft.py --num-samples -1 --gpu 4

    # Compare finetuned vs base model
    python eval_sft.py --num-samples 200 --gpu 4 --compare-base
"""

import argparse
import json
import os
import re
import sys
import random
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
LLAVA_DIR = ROOT_DIR / "LLaVA"
sys.path.insert(0, str(LLAVA_DIR))

SFT_CHECKPOINT = ROOT_DIR / "checkpoints" / "llava-v1.6-mistral-7b-alf-sft"
BASE_MODEL = "liuhaotian/llava-v1.6-mistral-7b"
DATA_JSON = ROOT_DIR / "sft_data" / "alfworld-gpt4-45k.json"
IMAGE_FOLDER = ROOT_DIR / "sft_data" / "alf_data_folder"


# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_action_from_json_str(text: str) -> str | None:
    """Extract the 'action' value from a (possibly malformed) JSON response."""
    # Try strict JSON first
    try:
        obj = json.loads(text)
        return obj.get("action", "").strip()
    except json.JSONDecodeError:
        pass
    # Regex fallback for common model outputs
    m = re.search(r'"action"\s*:\s*"([^"]*)"', text)
    if m:
        return m.group(1).strip()
    return None


def load_model(model_path: str, device: str = "cuda", use_flash_attn: bool = True):
    """Load LLaVA-Mistral model + tokenizer + image processor."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map=device,
        use_flash_attn=use_flash_attn,
    )
    model.eval()
    return tokenizer, model, image_processor, context_len


def generate_response(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """Run inference on a single (image, prompt) pair and return the generated text."""
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava import conversation as conversation_lib

    # Build the conversation using the v1 (vicuna) template
    conv = conversation_lib.conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # Process image
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(device, dtype=model.dtype)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # LLaVA's generate() replaces the <image> token with image patch
    # embeddings internally, so output_ids length ≠ input_ids length.
    # Decode the full output and extract the assistant's response.
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # The v1 (vicuna) template separates with "ASSISTANT:"
    if "ASSISTANT:" in full_text:
        generated = full_text.split("ASSISTANT:")[-1].strip()
    else:
        generated = full_text
    return generated


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(
    model,
    tokenizer,
    image_processor,
    samples: list[dict],
    image_folder: Path,
    device: str = "cuda",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """Run evaluation over a list of samples and return metrics."""
    results = []
    exact_action_match = 0
    valid_json_count = 0
    total = len(samples)

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        image_path = image_folder / sample["image"]
        if not image_path.exists():
            tqdm.write(f"[SKIP] Image not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        # The human turn contains the full prompt (with <image> token)
        prompt = sample["conversations"][0]["value"]
        gt_response = sample["conversations"][1]["value"]

        # Ground-truth action
        gt_action = parse_action_from_json_str(gt_response)

        # Generate prediction
        try:
            pred_response = generate_response(
                model, tokenizer, image_processor, image, prompt,
                max_new_tokens=max_new_tokens, temperature=temperature,
                device=device,
            )
        except Exception as e:
            tqdm.write(f"[ERROR] Sample {sample['id']}: {e}")
            pred_response = ""

        pred_action = parse_action_from_json_str(pred_response)

        # Metrics
        is_valid_json = pred_action is not None
        is_exact_match = (
            pred_action is not None
            and gt_action is not None
            and pred_action.lower().strip() == gt_action.lower().strip()
        )

        if is_valid_json:
            valid_json_count += 1
        if is_exact_match:
            exact_action_match += 1

        result = {
            "id": sample["id"],
            "task": sample.get("task", ""),
            "gt_action": gt_action,
            "pred_action": pred_action,
            "exact_match": is_exact_match,
            "valid_json": is_valid_json,
        }
        results.append(result)

        # Print periodic progress
        if (i + 1) % 20 == 0 or (i + 1) == total:
            acc = exact_action_match / (i + 1) * 100
            json_rate = valid_json_count / (i + 1) * 100
            tqdm.write(
                f"  [{i+1}/{total}] Action Accuracy: {acc:.1f}%  |  "
                f"Valid JSON: {json_rate:.1f}%"
            )

    evaluated = len(results)
    metrics = {
        "total_samples": total,
        "evaluated": evaluated,
        "exact_action_match": exact_action_match,
        "action_accuracy": exact_action_match / evaluated * 100 if evaluated else 0,
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_count / evaluated * 100 if evaluated else 0,
    }

    # Per-task breakdown
    task_stats: dict[str, dict] = {}
    for r in results:
        t = r["task"] or "unknown"
        if t not in task_stats:
            task_stats[t] = {"total": 0, "correct": 0}
        task_stats[t]["total"] += 1
        if r["exact_match"]:
            task_stats[t]["correct"] += 1
    for t in task_stats:
        s = task_stats[t]
        s["accuracy"] = s["correct"] / s["total"] * 100 if s["total"] else 0
    metrics["per_task"] = task_stats

    return {"metrics": metrics, "results": results}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT-finetuned LLaVA on ALFWorld")
    parser.add_argument("--model-path", type=str, default=str(SFT_CHECKPOINT),
                        help="Path to the finetuned checkpoint")
    parser.add_argument("--data-path", type=str, default=str(DATA_JSON))
    parser.add_argument("--image-folder", type=str, default=str(IMAGE_FOLDER))
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of samples to evaluate (-1 = all)")
    parser.add_argument("--gpu", type=int, default=4,
                        help="Physical GPU index to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy, >0 = sampling")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path (default: auto)")
    parser.add_argument("--compare-base", action="store_true",
                        help="Also evaluate the base (unfinetuned) model for comparison")
    args = parser.parse_args()

    # GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    # Load data
    print(f"Loading data from {args.data_path} ...")
    with open(args.data_path) as f:
        all_data = json.load(f)
    print(f"  Total samples: {len(all_data)}")

    # Sample subset
    random.seed(args.seed)
    if args.num_samples > 0 and args.num_samples < len(all_data):
        samples = random.sample(all_data, args.num_samples)
    else:
        samples = all_data
    print(f"  Evaluating on: {len(samples)} samples (seed={args.seed})")

    # ── Finetuned model ──
    print(f"\n{'='*60}")
    print(f"Loading FINETUNED model from: {args.model_path}")
    print(f"{'='*60}")
    t0 = time.time()
    tokenizer, model, image_processor, ctx_len = load_model(
        args.model_path, device=device
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print(f"\nRunning evaluation ...")
    t0 = time.time()
    ft_output = evaluate(
        model, tokenizer, image_processor, samples, Path(args.image_folder),
        device=device, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    elapsed = time.time() - t0
    ft_metrics = ft_output["metrics"]

    print(f"\n{'='*60}")
    print(f"  FINETUNED MODEL RESULTS  ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Samples evaluated : {ft_metrics['evaluated']}")
    print(f"  Action Accuracy   : {ft_metrics['action_accuracy']:.2f}%  "
          f"({ft_metrics['exact_action_match']}/{ft_metrics['evaluated']})")
    print(f"  Valid JSON Rate   : {ft_metrics['valid_json_rate']:.2f}%  "
          f"({ft_metrics['valid_json_count']}/{ft_metrics['evaluated']})")
    print(f"\n  Per-task breakdown:")
    for task, stats in sorted(ft_metrics["per_task"].items()):
        print(f"    {task:40s}  {stats['accuracy']:5.1f}%  "
              f"({stats['correct']}/{stats['total']})")

    # Free GPU memory before base model
    del model, tokenizer, image_processor
    torch.cuda.empty_cache()

    # ── Base model (optional) ──
    base_output = None
    if args.compare_base:
        print(f"\n{'='*60}")
        print(f"Loading BASE model from: {BASE_MODEL}")
        print(f"{'='*60}")
        t0 = time.time()
        tokenizer, model, image_processor, ctx_len = load_model(
            BASE_MODEL, device=device
        )
        print(f"  Model loaded in {time.time()-t0:.1f}s")

        print(f"\nRunning evaluation ...")
        t0 = time.time()
        base_output = evaluate(
            model, tokenizer, image_processor, samples, Path(args.image_folder),
            device=device, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        elapsed = time.time() - t0
        base_metrics = base_output["metrics"]

        print(f"\n{'='*60}")
        print(f"  BASE MODEL RESULTS  ({elapsed:.0f}s)")
        print(f"{'='*60}")
        print(f"  Samples evaluated : {base_metrics['evaluated']}")
        print(f"  Action Accuracy   : {base_metrics['action_accuracy']:.2f}%  "
              f"({base_metrics['exact_action_match']}/{base_metrics['evaluated']})")
        print(f"  Valid JSON Rate   : {base_metrics['valid_json_rate']:.2f}%  "
              f"({base_metrics['valid_json_count']}/{base_metrics['evaluated']})")

        # Side-by-side comparison
        print(f"\n{'='*60}")
        print(f"  COMPARISON:  Finetuned vs Base")
        print(f"{'='*60}")
        print(f"  Action Accuracy:  {ft_metrics['action_accuracy']:.2f}%  vs  "
              f"{base_metrics['action_accuracy']:.2f}%  "
              f"(Δ {ft_metrics['action_accuracy'] - base_metrics['action_accuracy']:+.2f}%)")
        print(f"  Valid JSON Rate:  {ft_metrics['valid_json_rate']:.2f}%  vs  "
              f"{base_metrics['valid_json_rate']:.2f}%  "
              f"(Δ {ft_metrics['valid_json_rate'] - base_metrics['valid_json_rate']:+.2f}%)")

        del model, tokenizer, image_processor
        torch.cuda.empty_cache()

    # ── Save results ──
    if args.output is None:
        out_path = ROOT_DIR / f"eval_results_n{len(samples)}.json"
    else:
        out_path = Path(args.output)

    save_data = {
        "args": vars(args),
        "finetuned": ft_output,
    }
    if base_output is not None:
        save_data["base"] = base_output

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
