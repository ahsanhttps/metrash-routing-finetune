#!/usr/bin/env python3
"""Compare base vs fine-tuned Gemma router on tricky test questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel

from routing_config import (
    LORA_DIR,
    MODEL_NAME,
    TEST_QUESTIONS_PATH,
    decode_model_output,
    format_router_prompt,
    parse_department_tag,
    prepare_model_inputs,
)


def load_test_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_base_model(model_path: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def attach_lora(model, lora_path: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, lora_path)


@torch.inference_mode()
def predict_department(model, tokenizer, query: str, max_new_tokens: int = 32) -> tuple[str | None, str]:
    prompt = format_router_prompt(tokenizer, query)
    inputs = prepare_model_inputs(tokenizer, prompt, device=str(model.device))
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
    )
    generated = decode_model_output(tokenizer, output_ids[0, inputs["input_ids"].shape[1] :])
    return parse_department_tag(generated), generated.strip()


def evaluate_label(name: str, model, tokenizer, questions: list[dict]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    correct = 0

    for item in questions:
        expected = item["expected"].lower()
        predicted, raw = predict_department(model, tokenizer, item["query"])
        is_correct = predicted == expected
        correct += int(is_correct)
        rows.append(
            {
                "id": item.get("id"),
                "query": item["query"],
                "expected": expected,
                "predicted": predicted,
                "raw": raw,
                "correct": is_correct,
            }
        )

    summary = {
        "model": name,
        "total": len(questions),
        "correct": correct,
        "accuracy": round(correct / len(questions), 4) if questions else 0.0,
    }
    return rows, summary


def print_table(base_rows: list[dict], lora_rows: list[dict], base_summary: dict, lora_summary: dict) -> None:
    print("\n" + "=" * 120)
    print("GEMMA DEPARTMENT ROUTING COMPARISON")
    print("=" * 120)
    print(f"Base model accuracy : {base_summary['correct']}/{base_summary['total']} ({base_summary['accuracy']:.1%})")
    print(f"LoRA model accuracy : {lora_summary['correct']}/{lora_summary['total']} ({lora_summary['accuracy']:.1%})")
    print("-" * 120)
    print(f"{'ID':<4} {'Expected':<10} {'Base':<10} {'LoRA':<10} {'Base OK':<8} {'LoRA OK':<8} Query")
    print("-" * 120)

    for base, lora in zip(base_rows, lora_rows):
        print(
            f"{str(base.get('id', '')):<4} "
            f"{base['expected']:<10} "
            f"{str(base['predicted'] or '-'):<10} "
            f"{str(lora['predicted'] or '-'):<10} "
            f"{str(base['correct']):<8} "
            f"{str(lora['correct']):<8} "
            f"{base['query'][:70]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned Gemma routing models")
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--lora-path", default=LORA_DIR)
    parser.add_argument("--test-file", default=TEST_QUESTIONS_PATH)
    parser.add_argument("--save-json", default="results/model_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_test_questions(Path(args.test_file))

    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = load_base_model(args.base_model)

    base_rows, base_summary = evaluate_label("base", model, tokenizer, questions)

    print(f"Loading LoRA adapters from: {lora_path}")
    lora_model = attach_lora(model, str(lora_path))
    FastLanguageModel.for_inference(lora_model)

    lora_rows, lora_summary = evaluate_label("lora", lora_model, tokenizer, questions)
    print_table(base_rows, lora_rows, base_summary, lora_summary)

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"base": {"summary": base_summary, "rows": base_rows}, "lora": {"summary": lora_summary, "rows": lora_rows}},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved detailed results to {save_path}")


if __name__ == "__main__":
    main()
