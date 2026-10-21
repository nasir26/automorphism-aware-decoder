# Proposed GraphCrunch change: vertex colourings and the lifted group order

Patch: `coloured_twins.patch`, against `vaibh-pra/GraphCrunch` commit `0d96131`. It has not been
sent or applied upstream. Apply with `patch -p1 < coloured_twins.patch` inside a GraphCrunch clone.

## What it changes

1. `src/orbits_serial.py`
   - `twin_classes(n, adj, colour=None)`: with a colouring, twins must also share a colour.
   - `structure(..., colour=None)`: the quotient colour becomes (input colour, class size, clique type).
   - `structure` now returns `log10_K` and `log10_order` (= log10 |K| + log10 |A|), reading |A| from
     the solver's existing `Order:` line, which the front end did not parse before.
   - CLI: `--colours FILE` (one `vertex colour` pair per line, unlisted vertices get colour 0);
     `--raw` passes the same colouring to the uncontracted solve so the cross-check stays meaningful;
     the run prints the group order.
2. `tests/test_coloured_orbits.py`: brute-force validation with colours, including the group order.
3. `Makefile`: `make test` also runs the new test.

The C solver is unchanged; it already accepted a colour file and printed the order and generators.

## Why the theorem extends to coloured graphs

Let G carry a vertex colouring c and call u, v coloured twins when c(u) = c(v) and they are twins
in the sense of Prakash (2026), equation (1). Then:

- the transposition (u v) of coloured twins preserves adjacency (twins) and colour, so it lies in
  Aut(G, c); hence K = ∏ Sym(C) over coloured twin classes is a subgroup of Aut(G, c);
- coloured twinhood is an equivalence relation: the transitivity argument of Lemma 2.1 goes through
  because the composed transposition is again colour preserving;
- every g in Aut(G, c) maps coloured twin classes onto coloured twin classes, since it preserves
  neighbourhoods and colours; the induced map on the quotient preserves (c, |C|, τ(C)) and quotient
  adjacency, giving ρ: Aut(G, c) → A with kernel K;
- conversely, for σ in A any choice of bijections C → σ(C) defines an element of Aut(G, c), because
  classes mapped to each other have the same size, internal type and original colour (the proof of
  Lemma 3.1 with colours added).

So Aut(G, c) ≅ K ⋊ A, and the vertex and edge orbit lifting formulas of Theorem 4.1 apply unchanged.
Colouring the quotient by size and type alone, without the input colour, would allow A to map a
class of one colour onto a class of another, which is unsound; the tests include such a case.

## Validation

On the patched tree, on this host (2026-10-21):

```
python3 tests/test_coloured_orbits.py   -> 1298 coloured graphs checked, 0 failures
python3 tests/test_orbits.py            -> 1258 graphs checked, 0 failures   (unchanged)
```

The same construction is implemented independently in this repository (`src/autsym/coloured.py`)
and validated against brute force on every coloured graph with up to 5 vertices (2 colours at 5
vertices, 3 colours up to 4) and further random and blow-up cases (`tests/test_coloured.py`).
