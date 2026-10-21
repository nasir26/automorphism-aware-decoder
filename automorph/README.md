# Automorph exports

Edge lists in the upload format of Automorph (automorph-frontend.vercel.app, SeneQa Computing): a
header line with the number of nodes and the number of edges, then one zero-indexed `u v` pair per
line. Written by `bench/export_graphs.py`.

For every swept configuration (`<family>_d<d>_r<d>_<basis>_p0.003`):

| File | Graph |
|---|---|
| `<tag>_decoding.txt` | the decoding graph: detectors `0..n_det-1`, boundary node `n_det` |
| `<tag>_weighted_subdivided.txt` | the same graph with a midpoint vertex on every edge, so edge weights can be carried as vertex colours |
| `<tag>_*.colours.txt` | one integer per vertex, line i is the colour of vertex i |

Colours: 0 for detectors, 1 for the boundary node, and in the subdivided graphs 2 + k for the
midpoint of an edge whose weight falls in bucket k (weights rounded to 9 significant digits, buckets
numbered in increasing weight).

## Proposed colour file for Automorph

Automorph accepts no colours today, so a run on these edge lists returns the uncoloured group (for
the subdivided graphs, the group of the bare graph with detectors and midpoints interchangeable only
where degrees allow). The proposal is an optional second upload with one non-negative integer per
line, line i the colour of vertex i, interpreted as nauty's partition (vertices may only be mapped
onto vertices of the same colour). nauty supports this directly through `lab`/`ptn` with
`defaultptn = FALSE`, as GraphCrunch's solver and this repository's `tools/nauty_aut.c` do.
