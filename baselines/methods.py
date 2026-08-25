# -*- coding: utf-8 -*-
"""Baseline method registry.

This file is the single reviewer-facing registry for all baselines. The
lightweight TopK and Greedy baselines call one shared C++ backend, while
SketchRefine and Progressive Shading call their Python solver ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict
import sys

import numpy as np

from solver_progressive_shading import detect as detect_progressive_shading
from solver_sketchrefine import detect as detect_sketchrefine


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    detect: Callable
    id: int


def _load_cpp_extension():
    cpp_dir = Path(__file__).resolve().parent / "cpp_backend"
    if str(cpp_dir) not in sys.path:
        sys.path.insert(0, str(cpp_dir))
    try:
        import baseline_cpp_core  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The baseline C++ extension is not built. Run: "
            "cd baselines/cpp_backend && python setup.py build_ext --inplace"
        ) from exc
    return baseline_cpp_core


def _amounts(raw_window):
    return np.asarray([amount for _, amount in raw_window], dtype=np.float64)


def _call_cpp(function_name, raw_in, raw_out, trigger_amt, anchor_type,
              theta, eps, **kwargs):
    ext = _load_cpp_extension()
    fn = getattr(ext, function_name)
    in_amts = _amounts(raw_in)
    out_amts = _amounts(raw_out)
    if function_name.startswith("detect_topk"):
        return int(fn(in_amts, out_amts, float(trigger_amt), anchor_type,
                      float(theta), float(eps), int(kwargs.get("k", 8))))
    return int(fn(in_amts, out_amts, float(trigger_amt), anchor_type,
                  float(theta), float(eps)))


def detect_topk_value(raw_in, raw_out, trigger_amt, anchor_type,
                      theta, eps, **kwargs):
    return _call_cpp("detect_topk_value", raw_in, raw_out, trigger_amt,
                     anchor_type, theta, eps, **kwargs)


def detect_greedy_value(raw_in, raw_out, trigger_amt, anchor_type,
                        theta, eps, **kwargs):
    return _call_cpp("detect_greedy_value", raw_in, raw_out, trigger_amt,
                     anchor_type, theta, eps, **kwargs)


def detect_topk_ratio(raw_in, raw_out, trigger_amt, anchor_type,
                      theta, eps, **kwargs):
    return _call_cpp("detect_topk_ratio", raw_in, raw_out, trigger_amt,
                     anchor_type, theta, eps, **kwargs)


def detect_greedy_ratio(raw_in, raw_out, trigger_amt, anchor_type,
                        theta, eps, **kwargs):
    return _call_cpp("detect_greedy_ratio", raw_in, raw_out, trigger_amt,
                     anchor_type, theta, eps, **kwargs)


def detect_greedy_fill(raw_in, raw_out, trigger_amt, anchor_type,
                       theta, eps, **kwargs):
    return _call_cpp("detect_greedy_fill", raw_in, raw_out, trigger_amt,
                     anchor_type, theta, eps, **kwargs)


_BASELINE_LIST = (
    BaselineSpec("topk_value", detect_topk_value, 1),
    BaselineSpec("greedy_value", detect_greedy_value, 2),
    BaselineSpec("topk_ratio", detect_topk_ratio, 5),
    BaselineSpec("greedy_ratio", detect_greedy_ratio, 6),
    BaselineSpec("greedy_fill", detect_greedy_fill, 7),
    BaselineSpec("sketchrefine", detect_sketchrefine, 10),
    BaselineSpec("progressive_shading", detect_progressive_shading, 11),
)

ALL_BASELINES: Dict[str, BaselineSpec] = {
    spec.name: spec for spec in _BASELINE_LIST
}
