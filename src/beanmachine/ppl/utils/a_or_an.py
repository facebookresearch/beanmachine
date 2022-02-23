#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re


def _first_word(s: str) -> str:
    r = re.search("\\w+", s)
    return r.group(0) if r else ""


_always_a = {"uniform"}
_always_an = {"18"}

_vowel_sounds = "aeiouxAEIOUX8"


def use_an(s: str) -> bool:
    w = _first_word(s)
    if len(w) == 0:
        return False
    if any(w.startswith(prefix) for prefix in _always_a):
        return False
    if any(w.startswith(prefix) for prefix in _always_an):
        return True
    return w[0] in _vowel_sounds


def a_or_an(s: str) -> str:
    return "an " + s if use_an(s) else "a " + s


def A_or_An(s: str) -> str:
    return "An " + s if use_an(s) else "A " + s
