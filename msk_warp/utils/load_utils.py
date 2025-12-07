import numpy as np
import warp as wp


def exclusive_scan(v, mark_empty: bool):
    result = [0] * (len(v) + 1)
    for i in range(1, len(result)):
        result[i] = result[i - 1] + v[i - 1]
    # Remove the last element to return the exclusive scan
    result = result[:-1]

    if mark_empty:
        for i in range(len(v)):
            if v[i] == 0:
                result[i] = -1

    return result


def to_warp_array(lst, dtype):
    arr = np.array(lst)
    # remove 2nd dimension if it exists
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(axis=1)
    return wp.from_numpy(arr, dtype=dtype)


def make_zero(shape, dtype):
    return wp.zeros(shape, dtype=dtype)


def make_full(val, shape, dtype):
    return wp.full(shape, val, dtype=dtype)
