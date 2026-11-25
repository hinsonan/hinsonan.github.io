/-!
# End-to-End Formally Verified Binary Search in Lean 4

Unlike an abstract math proof, this file verifies the **actual executable**
`binarySearch` implementation end-to-end:
1. **Termination:** Lean statically proves the search loop always terminates
   (no infinite loops from bad `lo`/`hi` updates).
2. **Memory safety:** Every array access `arr[mid]` is statically proven
   in-bounds at compile time.
3. **Soundness (`binarySearch_sound`):** If `binarySearch arr target = some idx`,
   then `idx < arr.size` and `arr[idx] = target` (even if `arr` isn't sorted!).
4. **Completeness (`binarySearch_complete`):** If `arr` is sorted and `target`
   exists anywhere in `arr`, `binarySearch` is guaranteed to return `some idx`
   (no off-by-one bugs).
-/

namespace BinarySearch

/-! ## 1. The Executable Implementation (~15 lines) -/

/-- Recursive binary search over the half-open range `[lo, hi)` of `arr`.
    Lean checks both `mid < arr.size` (bounds safety) and `hi - lo` strictly
    decreasing on every recursive call (termination). -/
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

/-- Binary search for `target` in `arr`. Returns `some idx` if found, `none` otherwise. -/
def binarySearch (arr : Array Nat) (target : Nat) : Option Nat :=
  searchLoop arr target 0 arr.size (Nat.le_refl _)

/-! ## 2. The Specification -/

/-- An array is sorted in non-decreasing order. -/
def Sorted (arr : Array Nat) : Prop :=
  ∀ (i j : Nat) (hi : i < arr.size) (hj : j < arr.size),
    i ≤ j → arr[i] ≤ arr[j]

/-! ## 3. Soundness Proof (If it returns `some idx`, `arr[idx] = target`) -/

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

/-- **Theorem (Soundness):**
    Whenever `binarySearch arr target` returns `some idx`, `idx` is guaranteed
    to be in-bounds and `arr[idx] = target`. -/
theorem binarySearch_sound (arr : Array Nat) (target idx : Nat)
    (hres : binarySearch arr target = some idx) :
    ∃ (hidx : idx < arr.size), arr[idx] = target :=
  searchLoop_sound arr target 0 arr.size (Nat.le_refl _) idx hres

/-! ## 4. Completeness Proof (If `arr` is sorted and contains `target`, it finds it) -/

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
        -- Since `arr` is sorted and `arr[mid] < target = arr[i]`, we must have `mid < i`.
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
        -- Since `arr[mid] > target = arr[i]` and `arr` is sorted, we must have `i < mid`.
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

/-- **Theorem (Completeness):**
    If `arr` is sorted and `target` appears at some index `i`, then
    `binarySearch arr target` is guaranteed to find a valid index. -/
theorem binarySearch_complete (arr : Array Nat) (target : Nat)
    (hsorted : Sorted arr)
    (i : Nat) (hi : i < arr.size) (htarget : arr[i] = target) :
    ∃ idx, binarySearch arr target = some idx :=
  searchLoop_complete arr target hsorted 0 arr.size (Nat.le_refl _) i hi
    (Nat.zero_le i) hi htarget

/-- **Corollary (Exactness on Sorted Arrays):**
    On a sorted array, `binarySearch` returns `none` if and only if
    `target` does not occur anywhere in `arr`. -/
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

end BinarySearch
