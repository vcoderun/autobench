from __future__ import annotations as _annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any

from pydantic import BaseModel

from autobench.data.datasets import Case
from autobench.errors import AutobenchError
from autobench.spec import BenchmarkSpec


class PydanticEvalsUnavailableError(AutobenchError):
    """Raised when the optional pydantic-evals runtime is requested but absent."""


class PydanticEvalCasePayload(BaseModel):
    name: str
    inputs: dict[str, Any]
    expected_output: dict[str, Any]
    metadata: dict[str, Any]


class PydanticEvalsDatasetPayload(BaseModel):
    name: str
    cases: list[PydanticEvalCasePayload]


class PydanticEvalsRuntime:
    def __init__(self, *, module_name: str = "pydantic_evals") -> None:
        self.module_name = module_name

    def is_available(self) -> bool:
        return importlib.util.find_spec(self.module_name) is not None

    def require_module(self) -> ModuleType:
        if not self.is_available():
            raise PydanticEvalsUnavailableError(
                f"Optional runtime '{self.module_name}' is not installed."
            )
        return importlib.import_module(self.module_name)

    def case_payload(self, case: Case) -> PydanticEvalCasePayload:
        return PydanticEvalCasePayload(
            name=case.id,
            inputs=case.input,
            expected_output=case.expected,
            metadata={"tags": case.tags, **case.metadata},
        )

    def dataset_payload(self, spec: BenchmarkSpec) -> PydanticEvalsDatasetPayload:
        dataset_name = spec.dataset.id or spec.benchmark.id
        return PydanticEvalsDatasetPayload(
            name=dataset_name,
            cases=[self.case_payload(case) for case in spec.dataset.cases],
        )


__all__ = (
    "PydanticEvalCasePayload",
    "PydanticEvalsDatasetPayload",
    "PydanticEvalsRuntime",
    "PydanticEvalsUnavailableError",
)
