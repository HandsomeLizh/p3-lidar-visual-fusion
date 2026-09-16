"""Small array grouping primitives shared by bounded map storage."""
import numpy as np


def unique_row_indices(rows):
    """Indices of first occurrences, in lexicographic row order.

    Integer lexsort avoids NumPy's structured-record comparator for axis=0
    unique. Its stable order retains the first point of each voxel exactly.
    """
    if not len(rows):return np.empty(0,dtype=np.int64)
    order=np.lexsort(rows.T[::-1])
    ordered=rows[order]
    return order[np.r_[True,np.any(ordered[1:]!=ordered[:-1],axis=1)]]
