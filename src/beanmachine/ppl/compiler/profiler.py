# Copyright (c) Facebook, Inc. and its affiliates.

import time
from typing import List


accumulate = "accumulate"
infer = "infer"
fix_problems = "fix_problems"
graph_infer = "graph_infer"
build_bmg_graph = "build_bmg_graph"
transpose_samples = "transpose_samples"
build_mcsamples = "build_mcsamples"


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


class ProfilerData:
    events: List[Event]
    in_flight: List[Event]

    def __init__(self) -> None:
        self.events = []
        self.in_flight = []

    def begin(self, kind: str) -> None:
        t = time.time_ns()
        e = Event(True, kind, t)
        self.events.append(e)
        self.in_flight.append(e)

    def finish(self, kind: str) -> None:
        t = time.time_ns()
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
