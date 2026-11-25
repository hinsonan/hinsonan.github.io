"""Python binary search translated directly from `BinarySearch.lean`.

Each function and invariant mirrors the Lean 4 specification:
- `search_loop` / `binary_search`: half-open interval `[lo, hi)` with
  `mid = lo + (hi - lo) // 2`, matching `BinarySearch.searchLoop`.
- `is_sorted`: runtime check corresponding to the Lean `Sorted` proposition.
- `verify_properties`: finite runtime checks mirroring `binarySearch_sound`,
  `binarySearch_complete`, and `binarySearch_none_iff` (unlike Lean, Python can
  only test finite examples rather than proving them for all possible arrays).
"""

from __future__ import annotations

from typing import Optional, Sequence


def search_loop(arr: Sequence[int], target: int, lo: int, hi: int) -> Optional[int]:
    """Recursive binary search over the half-open range `[lo, hi)`.

    Mirrors `BinarySearch.searchLoop` in `BinarySearch.lean`:
    - Precondition (proven statically in Lean): `0 <= lo` and `hi <= len(arr)`
    - Termination measure (proven in Lean): `hi - lo` strictly decreases
    """
    if lo < hi:
        mid = lo + (hi - lo) // 2
        val = arr[mid]
        if val == target:
            return mid
        if val < target:
            return search_loop(arr, target, mid + 1, hi)
        return search_loop(arr, target, lo, mid)
    return None


def binary_search(arr: Sequence[int], target: int) -> Optional[int]:
    """Binary search for `target` in `arr`. Returns index if found, else `None`.

    Mirrors `BinarySearch.binarySearch` in `BinarySearch.lean`.
    """
    return search_loop(arr, target, 0, len(arr))


def is_sorted(arr: Sequence[int]) -> bool:
    """Runtime equivalent of `BinarySearch.Sorted`."""
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


def verify_properties(arr: Sequence[int], target: int) -> None:
    """Spot-check the Lean theorems at runtime for a single `(arr, target)` pair."""
    res = binary_search(arr, target)

    # Mirrors `binarySearch_sound`
    if res is not None:
        assert 0 <= res < len(arr), "index out of bounds"
        assert arr[res] == target, "returned index does not hold target"

    # Mirrors `binarySearch_complete` & `binarySearch_none_iff`
    if is_sorted(arr):
        if res is None:
            assert all(x != target for x in arr), "missed an existing target"
        else:
            assert arr[res] == target


def main() -> None:
    arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
    queries = [2, 23, 91, 42, 0]

    print(f"Array: {arr}")
    for q in queries:
        verify_properties(arr, q)
        idx = binary_search(arr, q)
        if idx is not None:
            print(f"  search({q}) -> {idx} (arr[{idx}] = {arr[idx]})")
        else:
            print(f"  search({q}) -> None")

    # Sanity-check edge cases
    for edge_arr in [[], [7], [1, 1, 2, 2, 3]]:
        for q in [0, 1, 2, 3, 7, 99]:
            verify_properties(edge_arr, q)


if __name__ == "__main__":
    main()
