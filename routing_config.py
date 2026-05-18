"""Shared config for Qatar department routing with Gemma 2."""

from __future__ import annotations

import json
import re

DEPARTMENTS: tuple[str, ...] = ("visa", "residency", "traffic", "general")
COUNTRY = "Qatar"

MODEL_NAME = "unsloth/gemma-2-2b"
LORA_DIR = "gemma_router_lora"
DATASET_PATH = "data/routing_dataset.jsonl"
TEST_QUESTIONS_PATH = "test_questions.json"

ROUTER_INSTRUCTION = f"""You are a department routing classifier for a government service center in {COUNTRY}.
Read the user query in the input and respond with exactly one routing tag as JSON only.

Valid tags:
- visa: Qatar entry permits, visit/tourist/work/family visas, Hayya, visa renewal, cancellation, exit permit, overstay, Hamad International Airport entry, employer/kafeel sponsorship, Qatar Visa Center
- residency: Qatar ID (QID), residence permit (RP), Metrash2, civil registration, address change, tenancy contract attestation, fingerprinting, local ID cards
- traffic: Qatar driving license, vehicle registration (Istimara), MOI traffic fines, traffic violations, accident reports, vehicle transfer, Fahes inspection, parking violations
- general: greetings, office hours, contact info, fees without a specific service, walk-ins, Hukoomi portal help with no clear department, unrelated chit-chat

Rules:
1. Output ONLY valid JSON: {{"department": "<tag>"}}
2. Use lowercase tags only: visa, residency, traffic, general
3. If the query clearly maps to visa, residency, or traffic, NEVER use general
4. Use general only when no specific department can be inferred
5. Do not add explanations, markdown, or extra keys"""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_assistant_response(department: str) -> str:
    department = department.strip().lower()
    if department not in DEPARTMENTS:
        raise ValueError(f"Invalid department: {department}")
    return json.dumps({"department": department}, ensure_ascii=False)


def format_router_prompt(
    tokenizer,
    user_query: str,
    department: str | None = None,
    *,
    include_eos: bool = False,
) -> str:
    output = format_assistant_response(department) if department is not None else ""
    prompt = ALPACA_PROMPT.format(ROUTER_INSTRUCTION, user_query.strip(), output)
    if include_eos and department is not None:
        prompt += tokenizer.eos_token
    return prompt


def configure_tokenizer_for_generation(tokenizer) -> None:
    """Gemma needs a pad token for batch generation; avoid unset pad_token_id warnings."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def resolve_inference_device(model, device: str | None = None) -> str:
    if device is not None:
        return device
    return str(next(model.parameters()).device)


def prepare_model_inputs(tokenizer, prompt: str, device: str = "cuda"):
    configure_tokenizer_for_generation(tokenizer)
    return tokenizer([prompt], return_tensors="pt").to(device)


def decode_model_output(tokenizer, token_ids, skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def generate_router_completion(
    model,
    tokenizer,
    query: str,
    *,
    max_new_tokens: int = 32,
    min_new_tokens: int = 1,
    device: str | None = None,
) -> str:
    """Run greedy decoding for one routing query; returns only the new completion text."""
    device = resolve_inference_device(model, device)
    prompt = format_router_prompt(tokenizer, query)
    inputs = prepare_model_inputs(tokenizer, prompt, device=device)
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        max_length=None,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_token_ids = output_ids[0, input_len:]
    return decode_model_output(tokenizer, new_token_ids)


def parse_department_tag(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

    if "### Response:" in text:
        text = text.split("### Response:", 1)[-1].strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "department" in payload:
            tag = str(payload["department"]).strip().lower()
            return tag if tag in DEPARTMENTS else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*\"department\"\s*:\s*\"([a-z]+)\"[^{}]*\}", text, re.IGNORECASE)
    if match:
        tag = match.group(1).lower()
        return tag if tag in DEPARTMENTS else None

    lowered = text.lower()
    for tag in DEPARTMENTS:
        if re.search(rf"\b{re.escape(tag)}\b", lowered):
            return tag
    return None
