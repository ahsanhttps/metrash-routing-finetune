#!/usr/bin/env python3
"""Generate synthetic department-routing training data via Groq (OpenAI-compatible API)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from openai import OpenAI, RateLimitError

from routing_config import COUNTRY, DATASET_PATH, DEPARTMENTS, format_assistant_response

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = "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"

QATAR_CONTEXT = """
All examples must be set in Qatar and feel like real citizen/resident/expat queries to Qatari government services.

Use Qatar-specific context where natural, such as:
- MOI Qatar, Metrash2, Hukoomi, Qatar Visa Center, Hamad International Airport
- Qatar ID (QID), residence permit (RP), kafeel/sponsor, Hayya card
- Istimara, Fahes, MOI Traffic Department, driving license renewal in Qatar
- Doha, municipalities, expat workers, family visit, exit permit, overstay fines

Write in natural English (and occasional casual phrasing). Do not mention UAE, Emirates ID, Salik, or Ejari.
"""

DEPARTMENT_HINTS = {
    "visa": (
        "Qatar entry permits, visit/tourist/work/family visas, Hayya card, visa renewal, visa cancellation, "
        "exit permit, overstay fines, Hamad International Airport entry issues, employer/kafeel sponsorship, "
        "Qatar Visa Center, passport held for visa processing, change of employer visa transfer"
    ),
    "residency": (
        "Qatar ID (QID), residence permit (RP), Metrash2, civil registration, address change in Qatar, "
        "tenancy contract attestation, fingerprinting for QID, local ID card renewal, file number lookup, "
        "document resubmission for residency, family member RP addition"
    ),
    "traffic": (
        "Qatar driving license, vehicle registration (Istimara), MOI traffic fines, traffic violations, "
        "accident reports, vehicle ownership transfer, Fahes inspection, parking violations, traffic points, "
        "license renewal, plate number change"
    ),
    "general": (
        "greetings, MOI or service center office hours in Qatar, contact info, walk-ins, location in Doha, "
        "phone number, generic help requests with no specific visa/residency/traffic service, unrelated chit-chat, "
        "general Hukoomi portal questions without a clear department"
    ),
}

GENERATION_SYSTEM = f"""You generate training examples for a government service department router in {COUNTRY}.
{QATAR_CONTEXT}

Return ONLY a JSON array. Each item must be:
{{
  "query": "<natural user message>",
  "department": "<exact department provided by the user>"
}}

Rules:
- Every item in the array must use the exact department requested.
- Every query must clearly belong to Qatar government services context.
- Mix direct and indirect phrasing naturally.
- Keep queries short and realistic (1-2 sentences).
- Vary wording, tone, nationality, and scenario heavily. Avoid repeating patterns.
- Do not include markdown, commentary, or extra fields."""


class DatasetState:
    def __init__(self, rows: list[dict[str, str]], targets: dict[str, int]):
        self.lock = threading.Lock()
        self.rows = dedupe_rows(rows)
        self.targets = targets
        self.seen_queries: set[str] = set()
        self.query_index: dict[str, set[str]] = {dept: set() for dept in DEPARTMENTS}
        for row in self.rows:
            key = row["query"].strip().lower()
            self.seen_queries.add(key)
            self.query_index[row["department"]].add(key)
        self.counts = count_by_department(self.rows)

    def sample_existing(self, department: str, limit: int = 12) -> list[str]:
        with self.lock:
            pool = list(self.query_index[department])
        if not pool:
            return []
        return random.sample(pool, min(len(pool), limit))

    def missing_slots(self) -> list[tuple[str, int]]:
        with self.lock:
            missing: list[tuple[str, int]] = []
            for department, target in self.targets.items():
                need = target - self.counts[department]
                if need > 0:
                    missing.append((department, need))
            return missing

    def total_rows(self) -> int:
        with self.lock:
            return len(self.rows)

    def merge_batch(self, batch: list[dict[str, str]]) -> int:
        added = 0
        with self.lock:
            for row in batch:
                key = row["query"].strip().lower()
                if key in self.seen_queries:
                    continue
                self.seen_queries.add(key)
                self.query_index[row["department"]].add(key)
                self.rows.append(row)
                self.counts[row["department"]] += 1
                added += 1
        return added

    def snapshot_counts(self) -> Counter[str]:
        with self.lock:
            return Counter(self.counts)

    def all_rows(self) -> list[dict[str, str]]:
        with self.lock:
            return list(self.rows)


def get_client(api_key: str | None = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    key = api_key or GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Groq API key is missing.")
    return OpenAI(api_key=key, base_url=base_url)


def extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array found in model output: {text[:300]}")

    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array")
    return payload


def normalize_example(raw: dict[str, Any], expected_department: str | None = None) -> dict[str, str] | None:
    query = str(raw.get("query", "")).strip()
    department = str(raw.get("department", "")).strip().lower()

    if expected_department is not None:
        department = expected_department

    if not query or department not in DEPARTMENTS:
        return None

    return {
        "query": query,
        "department": department,
        "response": format_assistant_response(department),
    }


def build_batch_prompt(
    department: str,
    count: int,
    existing_samples: list[str],
) -> str:
    hint = DEPARTMENT_HINTS[department]
    sample_block = ""
    if existing_samples:
        joined = "\n".join(f"- {q}" for q in existing_samples[:8])
        sample_block = f"\nDo NOT repeat or closely paraphrase these existing queries:\n{joined}\n"

    return (
        f"Generate exactly {count} routing examples for {COUNTRY}.\n"
        f"Department for ALL items: {department}\n"
        f"Department scope in Qatar: {hint}\n"
        f"Use different personas (expat worker, Qatari citizen, visitor, family dependent), "
        f"Qatar locations, and sentence structures in every item."
        f"{sample_block}"
    )


def generate_batch_with_retries(
    *,
    model: str,
    department: str,
    count: int,
    temperature: float,
    existing_samples: list[str],
    max_retries: int,
) -> list[dict[str, str]]:
    client = get_client()
    last_error: Exception | None = None

    for retry in range(max_retries):
        try:
            user_prompt = build_batch_prompt(department, count, existing_samples)
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            rows = extract_json_array(content)

            cleaned: list[dict[str, str]] = []
            for row in rows:
                item = normalize_example(row, expected_department=department)
                if item is not None:
                    cleaned.append(item)
            if not cleaned:
                raise ValueError("Batch returned zero valid examples")
            return cleaned
        except RateLimitError as exc:
            last_error = exc
            time.sleep(min(60, 2 ** retry))
        except Exception as exc:
            last_error = exc
            time.sleep(min(20, 1 + retry * 2))

    raise RuntimeError(f"Batch failed for {department}: {last_error}")


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        key = row["query"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def load_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            normalized = normalize_example(item)
            if normalized is not None:
                rows.append(normalized)
    return dedupe_rows(rows)


def save_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_by_department(rows: list[dict[str, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[row["department"]] += 1
    return counter


def print_stats(rows: list[dict[str, str]]) -> None:
    by_department = count_by_department(rows)
    print(f"Total examples: {len(rows)}", flush=True)
    print(f"By department: {dict(by_department)}", flush=True)


def department_targets(per_department: int) -> dict[str, int]:
    return {department: per_department for department in DEPARTMENTS}


def build_jobs(state: DatasetState, batch_size: int, worker_count: int) -> list[tuple[str, int]]:
    missing = state.missing_slots()
    if not missing:
        return []

    random.shuffle(missing)
    jobs: list[tuple[str, int]] = []
    for department, need in missing:
        while need > 0 and len(jobs) < worker_count:
            count = min(batch_size, need)
            jobs.append((department, count))
            need -= count
        if len(jobs) >= worker_count:
            break
    return jobs


def run_batch_job(
    job: tuple[str, int],
    *,
    state: DatasetState,
    model: str,
    temperature: float,
    max_retries: int,
) -> tuple[str, int, list[dict[str, str]]]:
    department, count = job
    existing_samples = state.sample_existing(department)
    batch = generate_batch_with_retries(
        model=model,
        department=department,
        count=count,
        temperature=temperature,
        existing_samples=existing_samples,
        max_retries=max_retries,
    )
    return department, count, batch


def parse_args() -> argparse.Namespace:
    default_workers = os.cpu_count() or 4
    parser = argparse.ArgumentParser(description="Generate routing dataset with Groq")
    parser.add_argument("--output", default=DATASET_PATH, help="Output JSONL path")
    parser.add_argument("--seed", default="data/seed_routing_dataset.jsonl", help="Optional seed JSONL to merge")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model name")
    parser.add_argument(
        "--per-department",
        type=int,
        default=1000,
        help="Target examples per department (visa, residency, traffic, general)",
    )
    parser.add_argument("--batch-size", type=int, default=20, help="Examples requested per API call")
    parser.add_argument("--workers", type=int, default=default_workers, help="Concurrent API workers")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--max-retries", type=int, default=8, help="Retries per failed batch")
    parser.add_argument("--save-every", type=int, default=5, help="Save after this many completed batches")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--no-seed", action="store_true", help="Do not merge seed file")
    parser.add_argument("--api-key", default=None, help="Optional Groq API key override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.api_key:
        global GROQ_API_KEY
        GROQ_API_KEY = args.api_key

    output_path = Path(args.output)
    seed_path = Path(args.seed)
    targets = department_targets(args.per_department)
    total_target = args.per_department * len(DEPARTMENTS)

    if args.resume and output_path.exists():
        rows = load_existing(output_path)
        print(f"Resuming from {output_path} with {len(rows)} existing rows", flush=True)
    elif not args.no_seed and seed_path.exists():
        rows = load_existing(seed_path)
        print(f"Starting from seed file with {len(rows)} rows", flush=True)
    else:
        rows = []

    state = DatasetState(rows, targets)
    workers = max(1, args.workers)
    print(f"Target total: {total_target} ({args.per_department} per department)", flush=True)
    print(f"Using {workers} concurrent workers (cpu_count={os.cpu_count()})", flush=True)

    completed_batches = 0
    failed_batches = 0
    attempt_budget = 0
    max_attempt_budget = (total_target // max(1, args.batch_size) + 1) * 6

    with ThreadPoolExecutor(max_workers=workers) as executor:
        in_flight: set[Future] = set()

        while attempt_budget < max_attempt_budget:
            while len(in_flight) < workers:
                jobs = build_jobs(state, args.batch_size, workers - len(in_flight))
                if not jobs:
                    break
                for job in jobs:
                    future = executor.submit(
                        run_batch_job,
                        job,
                        state=state,
                        model=args.model,
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                    )
                    in_flight.add(future)

            if not in_flight:
                break

            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                attempt_budget += 1
                try:
                    department, requested, batch = future.result()
                    added = state.merge_batch(batch)
                    completed_batches += 1
                    counts = state.snapshot_counts()
                    print(
                        f"[{completed_batches}] +{added}/{requested} "
                        f"{department}={counts[department]}/{targets[department]} "
                        f"total={state.total_rows()}/{total_target}",
                        flush=True,
                    )
                    if completed_batches % args.save_every == 0:
                        save_jsonl(output_path, state.all_rows())
                except Exception as exc:
                    failed_batches += 1
                    print(f"Batch failed: {exc}", flush=True)

            if not state.missing_slots():
                break

    final_rows = state.all_rows()
    save_jsonl(output_path, final_rows)
    counts = state.snapshot_counts()

    print(f"\nSaved dataset to {output_path}", flush=True)
    print_stats(final_rows)
    for department in DEPARTMENTS:
        print(f"{department}: {counts[department]}/{args.per_department}", flush=True)

    print(f"Completed batches: {completed_batches}, failed batches: {failed_batches}", flush=True)
    if len(final_rows) < total_target:
        print(
            f"Warning: generated {len(final_rows)}/{total_target}. Re-run with --resume to continue.",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
