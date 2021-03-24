# Copyright (c) Facebook, Inc. and its affiliates.
import json as json_
from typing import Any, Optional

from beanmachine.ppl.compiler.profiler import event_list_to_report


class PerformanceReport:
    json: Optional[str] = None
    # TODO: Add a __str__ function


def _to_perf_rep(v: Any) -> Any:
    if isinstance(v, dict):
        p = PerformanceReport()
        for key, value in v.items():
            value = _to_perf_rep(value)
            setattr(p, key, value)
        return p
    if isinstance(v, list):
        for i in range(len(v)):
            v[i] = _to_perf_rep(v[i])
        return v
    return v


def json_to_perf_report(json: str) -> PerformanceReport:
    d = json_.loads(json)
    p = _to_perf_rep(d)
    assert isinstance(p, PerformanceReport)
    p.json = json
    if hasattr(p, "profiler_data"):
        # pyre-ignore
        p.bmg_profiler_report = event_list_to_report(p.profiler_data)
    return p
