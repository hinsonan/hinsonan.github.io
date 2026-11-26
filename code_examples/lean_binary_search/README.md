# End-to-End Formally Verified Binary Search in Lean 4

Companion Lean 4 code for the blog post **Getting Shredded with Lean**.

## What Is Proven (`BinarySearch.lean`)

The executable `binarySearch` implementation (~15 lines) is verified **end-to-end** by the Lean 4 kernel with **zero `sorry`s** and **zero external dependencies**:

1. **Termination & Bounds Safety:** Proven statically inside `searchLoop` (`termination_by hi - lo` and `mid < arr.size`).
2. `binarySearch_sound`: If `binarySearch arr target = some idx`, then `idx < arr.size` and `arr[idx] = target`.
3. `binarySearch_complete`: If `arr` is sorted and `target` is in `arr`, `binarySearch` is guaranteed to return `some idx`.
4. `binarySearch_none_iff`: On a sorted array, `binarySearch` returns `none` if and only if `target` does not occur anywhere in `arr`.

## Build & Run

From this directory:

```bash
lake build             # Type-check and verify all theorems
lake exe binary_search # Run the executable Lean binary search demo
python3 binary_search.py # Run the Python implementation translated from the Lean spec
```
