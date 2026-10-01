# Baseline C++ Core

This directory contains the C++17 pybind11 implementation of the lightweight
baseline decision routines:

- `query_topk_value`
- `query_topk_ratio`
- `query_greedy_value`
- `query_greedy_ratio`
- `query_greedy_fill`

Build:

```bash
python setup.py build_ext --inplace
```

`../methods.py` converts streaming window data into NumPy arrays and dispatches
the decision step to this extension.
