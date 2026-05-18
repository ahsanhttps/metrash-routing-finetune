"""Shared config for department routing fine-tune, dataset generation, and evaluation."""

from __future__ import annotations

import json
import re
from typing import Any

DEPARTMENTS: tuple[str, ...] = ("visa", "residency", "traffic", "general")
COUNTRY = "Qatar"

SYSTEM_PROMPT = f"""You are a department routing classifier for a government service center in {COUNTRY}.
Your job is to read the user query and output exactly one routing tag.

Valid tags:
- visa: Qatar entry permits, visit/tourist/work/family visas, Hayya, visa renewal, cancellation, exit permit, overstay, Hamad International Airport entry, employer/kafeel sponsorship, Qatar Visa Center
- residency: Qatar ID (QID), residence permit (RP), Metrash2, civil registration, address change, tenancy contract attestation, fingerprinting, local ID cards
- traffic: Qatar driving license, vehicle registration (Istimara), MOI traffic fines, traffic violations, accident reports, vehicle transfer, Fahes inspection, parking violations
- general: greetings, office hours, contact info, fees without a specific service, walk-ins, Hukoomi portal help with no clear department, unrelated chit-chat

Rules:
1. Output ONLY valid JSON with this exact schema: {{"department": "<tag>"}}
2. Use lowercase tags only: visa, residency, traffic, general
3. If the query clearly maps to visa, residency, or traffic, NEVER use general
4. Use general only when no specific department can be inferred
5. Do not add explanations, markdown, or extra keys"""

MODEL_NAME = "unsloth/Qwen3.5-0.8B"
LORA_DIR = "qwen_router_lora"
DATASET_PATH = "data/routing_dataset.jsonl"
TEST_QUESTIONS_PATH = "test_questions.json"


def format_assistant_response(department: str) -> str:
    department = department.strip().lower()
    if department not in DEPARTMENTS:
        raise ValueError(f"Invalid department: {department}")
    return json.dumps({"department": department}, ensure_ascii=False)


def build_messages(user_query: str, department: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query.strip()},
    ]
    if department is not None:
        messages.append({"role": "assistant", "content": format_assistant_response(department)})
    return messages


def parse_department_tag(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

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
