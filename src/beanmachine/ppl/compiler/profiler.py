# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Dict, List, Optional


accumulate = "accumulate"
infer = "infer"
fix_problems = "fix_problems"
graph_infer = "graph_infer"
build_bmg_graph = "build_bmg_graph"
transpose_samples = "transpose_samples"
build_mcsamples = "build_mcsamples"
deserialize_perf_report = "deserialize_perf_report"


class Event:
    begin: bool
    kind: str
    timestamp: int

    def __init__(self, begin: bool, kind: str, timestamp: int) -> None:
        self.begin = begin
        self.kind = kind
        self.timestamp = timestamp

    def __str__(self) -> str:
        s = "begin" if self.begin else "finish"
        return f"{s} {self.kind} {self.timestamp}"


class ProfileReport:
    calls: int
    total_time: int
    children: Dict[str, "ProfileReport"]
    parent: Optional["ProfileReport"]

    def __init__(self) -> None:
        self.calls = 0
        self.total_time = 0
        self.children = {}
        self.parent = None

    def _to_string(self, indent: str) -> str:
        s = ""
        attributed = 0
        # TODO: Sort by total time of children - WARNING: Important not to sort by
        #       runtime, as this leaks timing non-determinsm into report structure.
        # TODO: compute unattributed via property
        for key, value in self.children.items():
            s += f"{indent}{key}:({value.calls}) {value.total_time // 1000000} ms\n"
            s += value._to_string(indent + "  ")
            attributed += value.total_time
        # Commenting the "anded" part out to limit leakage of timing non-determinsim
        # TODO: There are two shortcomings to the current solution. First, it prints
        # a somewhat confusing final "unattributed" label that (currently) represents
        # the total time. Second, even in other positions, it is not ideal to print
        # only the absolute value of the unattributed time, as the reader is likely
        # to assume a direction for the mismatch. A better way to do this would be
        # change the sanitizing printer to fix this.
        if len(self.children) > 0:  # and self.total_time > 0:
            unattributed = self.total_time - attributed
            s += f"{indent}unattributed: {abs(unattributed // 1000000)} ms\n"
        return s

    def __str__(self) -> str:
        return self._to_string("")


class ProfilerData:
    events: List[Event]
    in_flight: List[Event]

    def __init__(self) -> None:
        self.events = []
        self.in_flight = []

    def begin(self, kind: str, timestamp: Optional[int] = None) -> None:

        t = time.time_ns() if timestamp is None else timestamp
        e = Event(True, kind, t)
        self.events.append(e)
        self.in_flight.append(e)

    def finish(self, kind: str, timestamp: Optional[int] = None) -> None:
        t = time.time_ns() if timestamp is None else timestamp
        while len(self.in_flight) > 0:
            top = self.in_flight.pop()
            e = Event(False, top.kind, t)
            self.events.append(e)
            if top.kind == kind:
                break

    def __str__(self) -> str:
        return "\n".join(str(e) for e in self.events)

    def time_in(self, kind: str) -> int:
        total_time = 0
        nesting = 0
        outermost_begin = None

        for e in self.events:
            if e.kind != kind:
                continue
            if nesting == 0 and e.begin:
                # We've found an outermost begin event.
                outermost_begin = e
                nesting = 1
            elif nesting == 1 and not e.begin:
                # We've found an outermost finish event
                nesting = 0
                assert isinstance(outermost_begin, Event)
                total_time += e.timestamp - outermost_begin.timestamp
                outermost_begin = None
            elif nesting > 0:
                # We've found a nested begin or finish
                if e.begin:
                    nesting += 1
                else:
                    nesting -= 1
        return total_time

    def to_report(self) -> ProfileReport:
        return event_list_to_report(self.events)


def event_list_to_report(events) -> ProfileReport:
    root = ProfileReport()
    current = root
    begins = []
    for e in events:
        if e.begin:
            if e.kind in current.children:
                p = current.children[e.kind]
            else:
                p = ProfileReport()
                p.parent = current
                current.children[e.kind] = p
                setattr(current, e.kind, p)
            p.calls += 1
            current = p
            begins.append(e)
        else:
            assert len(begins) > 0
            b = begins[-1]
            assert e.kind == b.kind
            current.total_time += e.timestamp - b.timestamp
            begins.pop()
            current = current.parent
    assert len(begins) == 0
    assert current == root
    return root
