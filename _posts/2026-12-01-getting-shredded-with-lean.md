---
layout: post
title: "Getting Shredded with Lean"
date: 2026-12-01
categories: ML
topic: Software Engineering
description: "Formal verification is all the rage, but does Lean actually keep coding agents in check?"
image: /assets/images/social/getting-shredded-with-lean.png
---

Lean is all the rage right now and the math community has finally emerged from their cave. Before AI started "solving" some of the hardest math problems on record nobody gave a crap about mathematicians and honestly there was very little engagement online within the STEM community. If you were not in their niche domain then you never heard anything from them.

Now there has been an explosion of videos and talks online from multiple mathematicians all the way from the most famous to the graduate student. As of recently the Navier-Stokes problem and the Riemann Hypothesis have been allegedly "solved". Now I say allegedly because the community is still verifying and gathering their information from these AI findings. Many of these problems have nuance to them and in many cases the important parts are how can these findings be applied/explained/used.

But at the heart of this is a programming language called [Lean](https://lean-lang.org/). Lean is a functional language and theorem prover built to formalize math theorems and provide formal verifications for theorems/functions. These LLMs have produced thousands of passing theorems written in Lean.

We will talk about the absolute frantic behavior of the math community in another post. This article is going to focus on Lean and formal verification applied to the field of software/ML/algorithms.

## The Problem

Coding agents are non-deterministic and using them to create large or complicated features can prove difficult at times. How do you maintain performance/architecture/features when agents just ripped a 10k git diff on your plate? How do you maintain control of the system and check your agents? Well since people are too scared to read the code we have to do something right?

## Formal Verification as the Solution

Formal verification takes many shapes and forms. For example a type system (a real type system not python types) is a form of verification. You know that `float16 x = 1.5` is a 16 bit float. The compiler will enforce this. If LLMs can "solve" these hard problems through verification with Lean can it be used in other areas?

Can Lean and other formal verification languages be used in software/ML/algorithm development? Let's attempt.

## Binary Search in Lean

Let's rip some Lean on binary search. Binary Search can have off-by-one bugs and infinite loops. Let's use Lean to help us make a valid algorithm. The code lives in `code_examples/lean_binary_search/`.

### 1. The Implementation

This function in Lean gets compiled into C and then a binary. Here `(hhi : hi ≤ arr.size)` and `have hmid : mid < arr.size := by omega` show at compile time that `arr[mid]` can never go out of bounds. The proofs for this live in `Prop`.

```lean
def searchLoop (arr : Array Nat) (target : Nat)
    (lo hi : Nat) (hhi : hi ≤ arr.size) : Option Nat :=
  if h : lo < hi then
    let mid := lo + (hi - lo) / 2
    have hmid : mid < arr.size := by omega
    let val := arr[mid]
    if val = target then
      some mid
    else if val < target then
      searchLoop arr target (mid + 1) hi hhi
    else
      searchLoop arr target lo mid (by omega)
  else
    none
termination_by hi - lo

def binarySearch (arr : Array Nat) (target : Nat) : Option Nat :=
  searchLoop arr target 0 arr.size (Nat.le_refl _)
```

### 2. Defining What "Sorted" Means

This `Prop` does not return a bool but states that every pair of valid indices is sorted and itemA is less than or equal to itemB.

```lean
def Sorted (arr : Array Nat) : Prop :=
  ∀ (i j : Nat) (hi : i < arr.size) (hj : j < arr.size),
    i ≤ j → arr[i] ≤ arr[j]
```

### 3. Proving Soundness (No False Positives)

This proves that the search is sound (if it returns an index, that index is in-bounds and actually holds the target), and tactics like `omega`, `cases`, and `exact` generate a low-level proof term that Lean's kernel verifies from scratch.

```lean
theorem searchLoop_sound (arr : Array Nat) (target : Nat)
    (lo hi : Nat) (hhi : hi ≤ arr.size) (idx : Nat)
    (hres : searchLoop arr target lo hi hhi = some idx) :
    ∃ (hidx : idx < arr.size), arr[idx] = target := by
  unfold searchLoop at hres
  split at hres
  · rename_i hlt
    dsimp only at hres
    split at hres
    · rename_i heq
      cases hres
      exact ⟨by omega, heq⟩
    · split at hres
      · exact searchLoop_sound arr target _ hi hhi idx hres
      · exact searchLoop_sound arr target lo _ (by omega) idx hres
  · cases hres
termination_by hi - lo

theorem binarySearch_sound (arr : Array Nat) (target idx : Nat)
    (hres : binarySearch arr target = some idx) :
    ∃ (hidx : idx < arr.size), arr[idx] = target :=
  searchLoop_sound arr target 0 arr.size (Nat.le_refl _) idx hres
```

### 4. Proving Completeness (No Off-By-One Misses)

Proving the algorithm is sound is not enough. You could just return `none` and that would be valid. We also need to make sure the algorithm is complete. This code proves that if the array is sorted and the target is in it, the search will actually find it.

```lean
theorem searchLoop_complete (arr : Array Nat) (target : Nat)
    (hsorted : Sorted arr)
    (lo hi : Nat) (hhi : hi ≤ arr.size)
    (i : Nat) (hi_lt : i < arr.size)
    (hlo_le : lo ≤ i) (hi_gt : i < hi)
    (htarget : arr[i] = target) :
    ∃ idx, searchLoop arr target lo hi hhi = some idx := by
  unfold searchLoop
  split
  · dsimp only
    split
    · exact ⟨lo + (hi - lo) / 2, rfl⟩
    · rename_i hne
      split
      · rename_i hless
        have hmid_lt_i : lo + (hi - lo) / 2 < i := by
          cases Nat.lt_or_ge (lo + (hi - lo) / 2) i with
          | inl h => exact h
          | inr h =>
            have hval : arr[i] ≤ arr[lo + (hi - lo) / 2] :=
              hsorted i (lo + (hi - lo) / 2) hi_lt (by omega) h
            omega
        exact searchLoop_complete arr target hsorted (lo + (hi - lo) / 2 + 1) hi hhi i hi_lt
          (by omega) hi_gt htarget
      · rename_i hnotless
        have hi_lt_mid : i < lo + (hi - lo) / 2 := by
          cases Nat.lt_or_ge i (lo + (hi - lo) / 2) with
          | inl h => exact h
          | inr h =>
            have hval : arr[lo + (hi - lo) / 2] ≤ arr[i] :=
              hsorted (lo + (hi - lo) / 2) i (by omega) hi_lt h
            omega
        exact searchLoop_complete arr target hsorted lo (lo + (hi - lo) / 2) (by omega) i hi_lt
          hlo_le hi_lt_mid htarget
  · omega
termination_by hi - lo

theorem binarySearch_complete (arr : Array Nat) (target : Nat)
    (hsorted : Sorted arr)
    (i : Nat) (hi : i < arr.size) (htarget : arr[i] = target) :
    ∃ idx, binarySearch arr target = some idx :=
  searchLoop_complete arr target hsorted 0 arr.size (Nat.le_refl _) i hi
    (Nat.zero_le i) hi htarget
```

### 5. Putting It Together (Exactness)

Now we can combine the soundness and completeness proofs into an exactness theorem (`binarySearch_none_iff`), which proves `binarySearch` returns `none` if and only if the target is not in the array. The original `searchLoop` was pretty short, but we ended up with all these extra lines of code for the theorems. As you can see, even for something simple like this you get a lot of "bloat".

```lean
theorem binarySearch_none_iff (arr : Array Nat) (target : Nat)
    (hsorted : Sorted arr) :
    binarySearch arr target = none ↔ ∀ (i : Nat) (hi : i < arr.size), arr[i] ≠ target := by
  constructor
  · intro hnone i hi htarget
    obtain ⟨idx, hsome⟩ := binarySearch_complete arr target hsorted i hi htarget
    rw [hnone] at hsome
    cases hsome
  · intro hnotfound
    cases hres : binarySearch arr target with
    | none => rfl
    | some idx =>
      obtain ⟨hidx, heq⟩ := binarySearch_sound arr target idx hres
      exact absurd heq (hnotfound idx hidx)
```

### Running the Search

```bash
cd code_examples/lean_binary_search
lake build
lake exe binary_search

# Output
Array: #[2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
  search(2) -> some 0 (arr[0] = 2)
  search(23) -> some 5 (arr[5] = 23)
  search(91) -> some 9 (arr[9] = 91)
  search(42) -> none
  search(0) -> none
```

### Translating into Python

Now that we have a formally verified binary search in Lean we can use agents to translate this into your tech stack of choice and reference the Lean against what they build. Here is the Python.

```python
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
```

## Did this solve our problem?

This code works and is correct so does this mean the process of using Lean is good enough for us to start letting coding agents rip huge features? Not at all and let's look at the flaws.

This code matches the spec and the spec only. In Lean, a tail-recursive function like `searchLoop` is compiled down to a flat C loop with `O(1)` stack space via tail-call optimization (TCO). The big issue is that Python has **no tail-call optimization**, so our direct 1-to-1 translation from Lean adds a new stack frame on every recursive step. For a linear recursive function this gives a `RecursionError`:

```lean
-- Lean: tail-recursive sum compiles to a flat O(1)-stack loop
def sumTail (xs : List Nat) (acc : Nat := 0) : Nat :=
  match xs with
  | []      => acc
  | x :: xs => sumTail xs (acc + x)
```

```python
# Python: direct 1-to-1 translation of the same tail recursion (no TCO!)
def sum_tail(xs: list[int], idx: int = 0, acc: int = 0) -> int:
    if idx == len(xs):
        return acc
    return sum_tail(xs, idx + 1, acc + xs[idx])  # Crashes with RecursionError at ~1,000 elements
```

To get the same `O(1)` stack behavior in Python for our binary search, we have to rewrite `search_loop` into an iterative `while` loop. Now we have drifted from the Lean spec. We are no longer 1-to-1 with Lean.

```python
def binary_search_iterative(arr: Sequence[int], target: int) -> Optional[int]:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        val = arr[mid]
        if val == target:
            return mid
        if val < target:
            lo = mid + 1
        else:
            hi = mid
    return None
```

This version of the code is much better when it comes to managing the constraints of Python and callstack memory. The Lean formal verification is still a good reference to use but it fell apart when met with Python and the "real" solution.

This just continues to get worse for other problems when translated to real systems. Lean does not prove intent and it also has no concept of problems relating to memory, concurrency, parallelism, etc...

Lean alone is not enough. You need to build in many other methods for formal verification.

## Conclusion

Lean and other languages like TLA+ are still helpful. They are great for thinking through your design. They force you to think before you command agents to pump out garbage. It also provides a reference check agents can use. This can help account for certain edge cases.

These are helpful tools to use but to be more effective you need other tools used in tandem (lint rules, static analysis, multi-stage agent reviews, etc...). Also you need to be reading your code and especially the difficult and critical parts. Reading the code is still the best way and you should be reading a lot of code.