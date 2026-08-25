# Baseline C++ Core

This directory contains the C++17 pybind11 implementation of the lightweight
baseline decision routines:

- `detect_topk_value`
- `detect_topk_ratio`
- `detect_greedy_value`
- `detect_greedy_ratio`
- `detect_greedy_fill`

Build:

```bash
python setup.py build_ext --inplace
```

`../methods.py` converts streaming window data into NumPy arrays and dispatches
the decision step to this extension.
