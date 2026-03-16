from typing import TypeVar

T = TypeVar("T")
V = TypeVar("V")


def string_list_to_ordering(l: list[str]) -> dict[str, int]:
    """ Converts a list of strings to a mapping from string to index in the list """
    return {v: i for i, v in enumerate(l)}


def reorder_list_for_ordering(l: list[str], ordering: dict[str, int]) -> list[int]:
    """ Returns the indices of the list l in the order specified by ordering """
    return [ordering[v] for v in l]


def apply_map_to_list(l: list[T], mapping: dict[T, V]) -> list[V]:
    """ Applies a mapping to a list """
    return [mapping[v] for v in l]


def gather(l: list[T], indices: list[int]) -> list[T]:
    """ Gathers the elements of a list at the specified indices """
    return [l[i] for i in indices]


def exclusive_scan(l: list[int]) -> list[int]:
    """ Returns the exclusive scan of a list, where the i-th element is the sum of all elements before i in the list """
    exclusive_sum_list = []
    running_sum = 0
    for v in l:
        exclusive_sum_list.append(running_sum)
        running_sum += v
    return exclusive_sum_list


def create_nested_list(l: list[T], num_per_sublist: int) -> list[list[T]]:
    """ Creates a nested list from a flat list, where each sublist has num_per_sublist elements """
    if len(l) % num_per_sublist != 0:
        raise ValueError(f"List length {len(l)} is not divisible by num_per_sublist {num_per_sublist}")

    nested_list = []
    for i in range(0, len(l), num_per_sublist):
        nested_list.append(l[i:i + num_per_sublist])
    return nested_list
