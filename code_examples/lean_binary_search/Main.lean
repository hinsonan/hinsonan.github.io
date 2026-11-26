import BinarySearch

open BinarySearch

def main : IO Unit := do
  let arr : Array Nat := #[2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
  let queries : List Nat := [2, 23, 91, 42, 0]
  IO.println s!"Array: {arr}"
  for q in queries do
    match binarySearch arr q with
    | some idx => IO.println s!"  search({q}) -> some {idx} (arr[{idx}] = {arr[idx]!})"
    | none     => IO.println s!"  search({q}) -> none"
