from __future__ import annotations
import json
from pathlib import Path


def main() -> None:
    path = Path("./outputs/aime2026/results.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    total = 0
    single_agent_correct = 0
    majority_voting_correct = 0
    scrd_correct = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            total += 1
            single_agent_correct += int(bool(item.get("single_agent_correct", False)))
            majority_voting_correct += int(bool(item.get("majority_voting_correct", False)))
            scrd_correct += int(bool(item.get("scrd_correct", False)))

    if total == 0:
        raise ValueError("results.jsonl is empty.")

    print(f"Total samples: {total}")
    print(f"Single-agent baseline accuracy: {single_agent_correct / total:.4f} ({single_agent_correct}/{total})")
    print(f"Majority-voting baseline accuracy: {majority_voting_correct / total:.4f} ({majority_voting_correct}/{total})")
    print(f"SCRD accuracy: {scrd_correct / total:.4f} ({scrd_correct}/{total})")


if __name__ == "__main__":
    main()