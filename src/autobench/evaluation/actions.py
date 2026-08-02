from __future__ import annotations as _annotations

from collections.abc import Iterable
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

from autobench.data.datasets import Case
from autobench.runtime.context import SpanKind, SpanRecord


class ExpectedAction(BaseModel):
    id: str
    kind: str = "tool"
    target: str
    input: Any = None
    output: Any = None
    order: int | None = None
    required: bool = True
    tolerance: dict[str, Any] = Field(default_factory=dict)


class ActionMatchResult(BaseModel):
    expected: ExpectedAction
    matched_span_id: str | None = None
    target_matched: bool = False
    input_matched: bool = False
    output_matched: bool = False

    @property
    def matched(self) -> bool:
        return self.target_matched and self.input_matched


ActionMetric: TypeAlias = Literal[
    "selection",
    "arguments",
    "sequence",
]


def expected_actions_from_case(case: Case) -> list[ExpectedAction]:
    expected = case.expected
    if not isinstance(expected, dict):
        return []
    raw_actions = expected.get("actions", expected.get("tool_calls", []))
    if not isinstance(raw_actions, list):
        return []
    actions: list[ExpectedAction] = []
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, dict):
            continue
        action_payload = dict(raw_action)
        action_payload.setdefault("id", f"action_{index + 1}")
        if "tool" in action_payload and "target" not in action_payload:
            action_payload["target"] = action_payload["tool"]
        if "args" in action_payload and "input" not in action_payload:
            action_payload["input"] = action_payload["args"]
        actions.append(ExpectedAction.model_validate(action_payload))
    return actions


def observed_action_spans(spans: list[SpanRecord], *, kind: str = "tool") -> list[SpanRecord]:
    return [
        span
        for span in spans
        if str(span.kind) == kind or (kind == "tool" and str(span.kind) == SpanKind.TOOL.value)
    ]


def match_expected_actions(
    expected_actions: list[ExpectedAction],
    observed_spans: list[SpanRecord],
) -> list[ActionMatchResult]:
    matches: list[ActionMatchResult] = []
    for action in expected_actions:
        match = _match_action(action, observed_spans)
        matches.append(match)
    return matches


def action_metric_score(
    expected_actions: list[ExpectedAction],
    observed_spans: list[SpanRecord],
    *,
    metric: ActionMetric,
) -> float:
    required_actions = [action for action in expected_actions if action.required]
    if not required_actions:
        return 1.0
    matches = match_expected_actions(required_actions, observed_spans)
    if metric == "selection":
        return _ratio(match.target_matched for match in matches)
    if metric == "arguments":
        return _ratio(match.input_matched for match in matches if match.target_matched)
    return 1.0 if _sequence_matches(required_actions, observed_spans) else 0.0


def _match_action(action: ExpectedAction, observed_spans: list[SpanRecord]) -> ActionMatchResult:
    for span in observed_spans:
        target_matched = (
            span.name == action.target or span.attributes.get("target") == action.target
        )
        if not target_matched:
            continue
        input_matched = _value_matches(action.input, span.input)
        output_matched = _value_matches(action.output, span.output)
        return ActionMatchResult(
            expected=action,
            matched_span_id=span.id,
            target_matched=True,
            input_matched=input_matched,
            output_matched=output_matched,
        )
    return ActionMatchResult(expected=action)


def _value_matches(expected: Any, observed: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key, expected_value in expected.items():
            if key not in observed:
                return False
            if not _value_matches(expected_value, observed[key]):
                return False
        return True
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) > len(observed):
            return False
        return all(
            _value_matches(expected_item, observed_item)
            for expected_item, observed_item in zip(expected, observed, strict=False)
        )
    return expected == observed


def _sequence_matches(
    expected_actions: list[ExpectedAction],
    observed_spans: list[SpanRecord],
) -> bool:
    ordered_expected = sorted(
        expected_actions,
        key=lambda action: action.order if action.order is not None else 0,
    )
    observed_targets = [span.attributes.get("target", span.name) for span in observed_spans]
    search_index = 0
    for action in ordered_expected:
        try:
            found_index = observed_targets.index(action.target, search_index)
        except ValueError:
            return False
        search_index = found_index + 1
    return True


def _ratio(values: Iterable[bool]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(1 for value in collected if value) / len(collected)


__all__ = (
    "ActionMatchResult",
    "ActionMetric",
    "ExpectedAction",
    "action_metric_score",
    "expected_actions_from_case",
    "match_expected_actions",
    "observed_action_spans",
)
