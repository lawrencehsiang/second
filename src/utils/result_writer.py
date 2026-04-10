from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultWriter:
    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.trace_dir = self.output_dir / "traces"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / "results.jsonl"
        self.errors_file = self.output_dir / "errors.jsonl"

    def append_result(self, result: dict[str, Any]) -> None:
        with self.results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def write_trace(self, sample_id: str, trace: list[dict[str, Any]]) -> None:
        trace_path = self.trace_dir / f"{sample_id}_trace.json"
        with trace_path.open("w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)

    def append_error(self, error_item: dict[str, Any]) -> None:
        with self.errors_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(error_item, ensure_ascii=False) + "\n")

    def load_completed_sample_ids(self) -> set[str]:
        completed: set[str] = set()
        if not self.results_file.exists():
            return completed

        with self.results_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = item.get("sample_id")
                if isinstance(sample_id, str) and sample_id:
                    completed.add(sample_id)
        return completed
