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

def check_zero(arr: wp.array):
    # if any the dimensions are zero, replace with a 1
    shape = list(arr.shape)
    found_zero = False
    for i in range(len(shape)):
        if shape[i] == 0:
            shape[i] = 1
            found_zero = True
    if not found_zero:
        return arr
    return wp.zeros(shape, dtype=arr.dtype)


def to_warp_array(lst, dtype):
    arr = np.array(lst)
    return check_zero(wp.from_numpy(arr, dtype=dtype))


def make_zero(shape, dtype):
    return check_zero(wp.zeros(shape, dtype=dtype))


def make_full(val, shape, dtype):
    return check_zero(wp.full(shape, val, dtype=dtype))
