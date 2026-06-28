"""Tiny CSV writer so every figure has its underlying data saved alongside it."""

import numpy as np


def save_table(path, columns):
    """Write a CSV from an ordered dict {column_name -> 1D array}.

    All columns must be numeric and the same length. Categorical dimensions
    should be encoded into column names (e.g. 'hub_mpsi') or numeric codes.
    """
    names = list(columns.keys())
    data = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    np.savetxt(path, data, delimiter=",", header=",".join(names), comments="")
    print(f"Saved {path}")
