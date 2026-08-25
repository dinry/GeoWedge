# -*- coding: utf-8 -*-
"""Python adapters for the C++17 GeoWedge extension."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from state_search import State, Txn


def _load_cpp_extension():
    cpp_dir = Path(__file__).resolve().parent / "cpp_bucket"
    if str(cpp_dir) not in sys.path:
        sys.path.insert(0, str(cpp_dir))
    try:
        import wedge_bucket_cpp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The C++ extension is not built. Run: "
            "cd geowedge/cpp_bucket && python setup.py build_ext --inplace"
        ) from exc
    return wedge_bucket_cpp


def frontier_search_cascade_cpp(
    transactions: List[Txn],
    trigger_eid: str,
    theta: float,
    eps: float,
    delta_sa: float = 0.1,
    delta_d: float = 0.1,
    max_states: int = 4000,
    compression_mode: str = "log_md",
) -> Optional[State]:
    """Run the C++ cascade search through the same Python pipeline signature."""
    ext = _load_cpp_extension()
    trigger = next(t for t in transactions if t.eid == trigger_eid)
    in_amts = np.asarray(
        [t.amount for t in transactions
         if t.eid != trigger_eid and t.direction == "in"],
        dtype=np.float64,
    )
    out_amts = np.asarray(
        [t.amount for t in transactions
         if t.eid != trigger_eid and t.direction == "out"],
        dtype=np.float64,
    )

    result = ext.frontier_search_cascade_compression_cpp(
        in_amts,
        out_amts,
        float(trigger.amount),
        trigger.direction,
        float(theta),
        float(eps),
        float(delta_sa),
        float(delta_d),
        int(max_states),
        compression_mode,
    )
    if result is None:
        return None
    sa, sb = result
    return State(sa=float(sa), sb=float(sb))
