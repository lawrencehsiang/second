from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class UsageRecord:
    sample_id: str | None
    round_id: int | None
    mode: str | None              # normal / repair / baseline / None
    component: str                # agent / recorder / evaluator / ...
    agent_id: str | None          # A / B / C / None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UsageLogger:
    """
    Stores call-level token usage records.

    Design goal:
    - Keep raw usage records first
    - Do aggregation later
    """

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    def log(
        self,
        *,
        sample_id: str | None,
        round_id: int | None,
        mode: str | None,
        component: str,
        agent_id: str | None,
        usage: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if usage is None:
            usage = {}

        record = UsageRecord(
            sample_id=sample_id,
            round_id=round_id,
            mode=mode,
            component=component,
            agent_id=agent_id,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            extra=extra or {},
        )
        self.records.append(record)

    def list_records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def sum_total_tokens(self) -> int:
        return sum(record.total_tokens for record in self.records)

    def sum_prompt_tokens(self) -> int:
        return sum(record.prompt_tokens for record in self.records)

    def sum_completion_tokens(self) -> int:
        return sum(record.completion_tokens for record in self.records)

    def sum_by_component(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for record in self.records:
            result[record.component] = result.get(record.component, 0) + record.total_tokens
        return result

    def sum_by_round(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for record in self.records:
            if record.round_id is None:
                continue
            result[record.round_id] = result.get(record.round_id, 0) + record.total_tokens
        return result

    def sum_by_mode(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for record in self.records:
            if record.mode is None:
                continue
            result[record.mode] = result.get(record.mode, 0) + record.total_tokens
        return result

    def filter_records(
        self,
        *,
        component: str | None = None,
        mode: str | None = None,
        round_id: int | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        result = []
        for record in self.records:
            if component is not None and record.component != component:
                continue
            if mode is not None and record.mode != mode:
                continue
            if round_id is not None and record.round_id != round_id:
                continue
            if agent_id is not None and record.agent_id != agent_id:
                continue
            result.append(record.to_dict())
        return result