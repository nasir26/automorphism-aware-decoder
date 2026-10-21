# Hypotheses

Copied from the governing prompt; the three approved changes of 2026-10-21 are marked (A). Each ends with a verdict: supported, refuted or
inconclusive. Further changes need your approval.

| Id | Statement | Verdict |
|---|---|---|
| H1 | Twin contraction of the fault graph reproduces the merged DEM exactly, at every tested d, both bases and every noise model tried, and preserves every automorphism of the coloured quotient. | pending (Phase 1) |
| H2 | With weights, the bulk group always contains time translation and σ, and the asymmetry is confined to a boundary layer of fixed depth (expected rounds 0, 1, T − 2, T − 1) independent of d. | pending (Phase 1) |
| H3 | At a fixed training budget, a σ-equivariant or σ-augmented pre-decoder beats the non-equivariant baseline on residual LER after PyMatching, and degrades less beyond its training distance. (A) Expected null: the decoding graph's group has order 2, so the experiment is powered for a small paired effect. | pending (Phase 3) |
| H4 | Orbit-indexed weight memory reduces on-chip storage by the orbit ratio (about 5.5x at d = 13), with bit-identical output and no timing or LUT penalty above an agreed threshold, at T3 then T2. | pending (Phase 5) |
| H5 | The σ-invariance test detects injected faults in an HLS decoder at a higher rate per test vector than random-vector comparison against PyMatching. (A) A fourth policy, weight-preserving relabelling (Shen and Zhong, arXiv:2609.07035), is compared too. | pending (Phase 6) |
| H6 | σ-twin disagreement improves the hierarchical decoder's accuracy against escalation trade-off when combined with the existing confidence gate. (A) Control: a second decode with randomly perturbed weights of equal cost, so a gain is attributed to the symmetry only if it beats the control. | pending (Phase 6) |
