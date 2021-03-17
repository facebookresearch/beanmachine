# Copyright (c) Facebook, Inc. and its affiliates.
import json as json_
from typing import Any, Dict, Optional


class PerformanceReport:
    json: Optional[str]

    def __init__(self, d: Optional[Dict[Any, Any]] = None, json: Optional[str] = None):
        self.json = json
        if json is not None:
            d = json_.loads(json)
        if d is not None:
            for key, value in d.items():
                # TODO: Recurse on dictionaries in lists also
                if isinstance(value, dict):
                    value = PerformanceReport(d=value)
                setattr(self, key, value)
