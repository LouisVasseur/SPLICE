import NFLProof.Gallop

/-! The same table as `harness.cpp`, computed by the Lean transcription `NFLProof.gallop`. -/

open NFLProof

def main : IO Unit := do
  for n in List.range 71 do
    if 1 ≤ n then
      for p in List.range n do
        for t in List.range n do
          let g := gallop (fun i => decide (i ≤ t)) p n
          IO.println s!"{n} {p} {t} {g.1} {g.2}"
