# -*- coding: utf-8 -*-
# Qwen3.5-0.8B Department Router — Colab training script
# Open in Colab: File → Upload notebook → select this .py file
# Or: https://colab.research.google.com → Upload → Qwen3_5_(0_8B)_Vision.py

# @markdown # Qwen3.5-0.8B Department Router (Text-Only JSON Tags)
# @markdown Fine-tune **Qwen3.5-0.8B** to route Qatar government queries into:
# @markdown - `visa`, `residency`, `traffic`, `general`
# @markdown
# @markdown Model output (routing tag only):
# @markdown ```json
# @markdown {"department": "visa"}
# @markdown ```
# @markdown
# @markdown **Before training:** upload `data/routing_dataset.jsonl`, `routing_config.py`, and `test_questions.json`
# @markdown or mount Google Drive below.


# @title 0. Colab setup (Drive + file check) { display-mode: "form" }
USE_GOOGLE_DRIVE = False  # @param {type:"boolean"}
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Unsloth"  # @param {type:"string"}

import os
import shutil
from pathlib import Path

if USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    src = Path(DRIVE_PROJECT_PATH)
    if src.exists():
        for name in ["routing_config.py", "test_questions.json", "data/routing_dataset.jsonl"]:
            src_file = src / name
            if src_file.exists():
                dest = Path("/content") / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest)
                print(f"Copied {src_file} → {dest}")
    else:
        print(f"Drive path not found: {src}")

required = [
    Path("routing_config.py"),
    Path("data/routing_dataset.jsonl"),
    Path("test_questions.json"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    print("Missing files:", missing)
    print("Upload them with the file picker or set USE_GOOGLE_DRIVE=True.")
else:
    print("All required files found.")


# @title 1. Installation
import importlib.util
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "-qqq", "uv"], check=False)

if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
    try:
        import numpy
        import PIL

        _numpy = f"numpy=={numpy.__version__}"
        _pil = f"pillow=={PIL.__version__}"
    except Exception:
        _numpy = "numpy"
        _pil = "pillow"
    subprocess.run(
        [
            "uv", "pip", "install", "-qqq",
            "torch==2.8.0", "triton>=3.3.0", _numpy, _pil,
            "bitsandbytes", "xformers==0.0.32.post2",
            "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
            "unsloth[base] @ git+https://github.com/unslothai/unsloth",
        ],
        check=False,
    )
elif importlib.util.find_spec("unsloth") is None:
    subprocess.run(["uv", "pip", "install", "-qqq", "unsloth"], check=False)

subprocess.run(
    [
        "uv", "pip", "install", "--upgrade", "--no-deps",
        "tokenizers>=0.22.0,<=0.23.0", "trl==0.22.2", "unsloth", "unsloth_zoo",
    ],
    check=False,
)
subprocess.run(["uv", "pip", "install", "transformers==5.2.0", "datasets"], check=False)
subprocess.run(
    ["uv", "pip", "install", "--no-build-isolation", "flash-linear-attention", "causal_conv1d==1.6.0"],
    check=False,
)
subprocess.run([sys.executable, "-m", "pip", "install", "-qqq", "openai"], check=False)
print("Installation complete.")


# @title 2. Load model
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-0.8B",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_16bit=True,
)


# @title 3. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
)


# @title 4. Data prep
import json
from pathlib import Path

from datasets import Dataset

from routing_config import build_messages

DATASET_PATH = Path("data/routing_dataset.jsonl")
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

print(f"Loading dataset from: {DATASET_PATH}")
raw_rows = []
with DATASET_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_rows.append(json.loads(line))


def to_conversation(row):
    department = row["department"].strip().lower()
    return {"messages": build_messages(row["query"], department)}


converted_dataset = [to_conversation(row) for row in raw_rows]
print(f"Examples: {len(converted_dataset)}")
print(converted_dataset[0])


def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        texts.append(text)
    return {"text": texts}


dataset = Dataset.from_list(converted_dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset[0]["text"][:500])


# @title 5. Pre-training inference check
from routing_config import parse_department_tag

FastLanguageModel.for_inference(model)

sample_query = "I want to apply for visa"
messages = build_messages(sample_query)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=32,
    use_cache=True,
    temperature=0.0,
    do_sample=False,
)
generated = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Raw:", generated)
print("Parsed tag:", parse_department_tag(generated))


# @title 6. Train
from trl import SFTConfig, SFTTrainer

FastLanguageModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=120,
        # num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_router",
        report_to="none",
        dataset_text_field="text",
    ),
)

trainer_stats = trainer.train()
trainer_stats.metrics


# @title 7. Inference helper
def route_query(query: str) -> dict:
    FastLanguageModel.for_inference(model)
    messages = build_messages(query)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        use_cache=True,
        temperature=0.0,
        do_sample=False,
    )
    raw = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    tag = parse_department_tag(raw)
    return {"department": tag, "raw": raw}


for q in [
    "I want to apply for visa",
    "What are your office hours?",
    "The camera on Salwa Road caught me yesterday.",
]:
    print(q, "->", route_query(q))


# @title 8. Evaluate on 15 tricky routing questions
with open("test_questions.json", "r", encoding="utf-8") as f:
    test_questions = json.load(f)

correct = 0
print(f"{'ID':<4} {'Expected':<10} {'Predicted':<10} OK   Query")
print("-" * 90)
for item in test_questions:
    result = route_query(item["query"])
    predicted = result["department"]
    ok = predicted == item["expected"]
    correct += int(ok)
    print(f"{item['id']:<4} {item['expected']:<10} {str(predicted):<10} {str(ok):<4} {item['query'][:55]}")

print(f"\nAccuracy: {correct}/{len(test_questions)} ({correct / len(test_questions):.1%})")


# @title 9. Save LoRA adapters
LORA_DIR = "qwen_router_lora"
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"Saved LoRA adapters to {LORA_DIR}")

if USE_GOOGLE_DRIVE:
    dest = Path(DRIVE_PROJECT_PATH) / LORA_DIR
    dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(LORA_DIR).exists():
        shutil.copytree(LORA_DIR, dest, dirs_exist_ok=True)
        print(f"Copied LoRA to Drive: {dest}")
