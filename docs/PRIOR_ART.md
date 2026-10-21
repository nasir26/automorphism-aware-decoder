# Prior art survey (Phase 0)

Survey date: 2026-10-21. Sources: arXiv abstract pages and full texts, search engine results.
Each entry states how far it was checked:

- **[full]** the full text, or the relevant sections of it, was read;
- **[abs]** the arXiv abstract page was read;
- **[snip]** only a search result snippet was seen; to be checked before it is cited in the paper.

For [snip] entries the titles and author lists are partly from memory and are unverified.
Nothing below is cited in the paper until it is at least [abs], and any specific number taken from
a work must be [full].

## 1. Symmetry and equivariance in neural decoders

| Work | What it exploits | Assumed or computed | Delta to this project |
|---|---|---|---|
| Egorov, Bondesan, Welling, *The END: an equivariant neural decoder for quantum error correction*, arXiv:2304.07362 (ICLR 2023 workshop) [snip] | translation equivariance of the optimal toric code decoder, twisted global average pooling | assumed from the lattice (toric code, periodic, code capacity) | no boundaries, no circuit-level noise, no reflection; a 2026 review reports that rotation, reflection and duality equivariance were left as future work [snip, arXiv:2608.15760] |
| Wagner, Kampermann, Bruß, *Symmetries for a high level neural decoder on the toric code*, arXiv:1910.01662 [snip] | translation symmetry by mapping each syndrome to a canonical representative | assumed (toric code) | preprocessing, not equivariant layers; toric, no time boundary |
| Weißl, Egorov, *Equivariant machine learning decoder for 3D toric codes*, arXiv:2409.04300 (QCNC 2025) [snip] | equivariance as inductive bias on the 3D toric code | assumed | periodic code, no circuit-level decoding graph |
| Wang, *group convolutional networks for toric codes*, AFIT thesis 2024 (scholar.afit.edu/etd/7696) [snip] | translation and rotation equivariance by group convolutions | assumed | reports no further gain with less exhaustive datasets; toric |
| Gu, Bonilla Ataides, Lukin, Yelin, *Scalable neural decoders for practical fault-tolerant quantum computation* (Cascade), arXiv:2604.08358, ICML 2026 [abs] | translation-equivariant convolution, adapted to the lattice symmetry of surface and bivariate bicycle codes | assumed from the code's lattice | boundary treatment not stated on the abstract page; no point reflection, no computed group |
| Chamberland, Olle, Li, Thornton, Baratta, *Fast and accurate AI-based pre-decoders for surface codes*, arXiv:2604.12841 (NVIDIA Ising) [snip; code read] | convolution (translation) in a 3D pre-decoder | assumed | the reference pre-decoder this project analyses; its tensor anchoring is not reflection symmetric (Phase 1 regression row) |
| Lee, Hur, Jae, Park, *Machine learning approaches to decoding topological quantum codes*, arXiv:2608.15760 (review) [abs] | survey | n/a | source for the statement that rotation and reflection equivariance remain open; that statement is [snip] and must be checked in the full text |

## 2. Automorphism ensemble decoding

| Work | What it exploits | Assumed or computed | Delta |
|---|---|---|---|
| Geiselhart, Elkelesh, Ebada, Cammerer, ten Brink, *Automorphism ensemble decoding of Reed–Muller codes*, arXiv:2012.07635 (author list and venue from memory, to be checked) [snip] | code automorphisms; parallel constituent decoders on permuted received words, outputs de-permuted, most likely kept | known algebraically (affine group of RM codes) | classical codes |
| Pillet, Bioglio, Land, *Polar codes for automorphism ensemble decoding*, arXiv:2102.08250 [snip] | designs polar codes whose automorphisms are not absorbed by SC decoding | algebraic | classical |
| Koutsioumpas, Sayginel, Webster, Browne, *Automorphism ensemble decoding of quantum LDPC codes* (AutDEC), arXiv:2503.01738 [full: method section and Table 1] | ensemble of BP decoders on syndromes transformed by automorphisms; for circuit-level noise, **automorphisms of the detector error model check matrix Tanner graph, computed with Bliss** (pyBliss); group orders 36 to 378 for bivariate bicycle codes | **computed from the circuit-level DEM** | the closest work to this project. Differences: BP on qLDPC codes, not matching on the surface code; no twin contraction; no account of how noise weights break the group or where; no use in neural pre-decoders, hardware memory or verification |

## 3. Ensembles and disagreement signals for matching decoders

| Work | What it exploits | Delta |
|---|---|---|
| Shutty, Newman, Villalonga, *Efficient near-optimal decoding of the surface code through ensembling* (harmonization), arXiv:2401.12434 [snip] | ensembles of correlated matching decoders with perturbed priors; a small ensemble flags high-risk shots for a larger one | the layered flagging idea is close to H6; their ensemble members differ by random prior perturbation, ours by a computed exact symmetry; must be cited against H6 |
| Jones, *Improved accuracy for decoding surface codes with matching synthesis* (Libra), arXiv:2408.12135 [snip] | combines solutions from an ensemble of perturbed matchers | ensemble baseline, not symmetry-derived |
| *Neural network decoder confidence as a learned proxy for the logical gap*, arXiv:2606.08758 [snip] | GNN soft output versus MWPM complementary gap for post-selection | confidence signal reference for H6 |

## 4. Symmetry in decoder parameters and hardware

| Work | What it exploits | Delta |
|---|---|---|
| Sivak, Newman, Klimov, *Optimization of decoder priors for accurate quantum error correction*, arXiv:2406.02700, PRL 133, 150603 (2024) [full: Section II and Appendix A] | **time-translation symmetry ties decoder edge parameters** (edges differing by a time shift share one parameter) to shrink a calibration problem | the time-translation part of H4's orbit table exists here as a calibration device; no spatial reflection, rule appears hand-derived, no hardware memory |
| Liyanage, Wu, Tagare, Zhong, *Helios* distributed Union-Find on FPGA, arXiv:2301.08419 and arXiv:2406.08491 (IEEE TQE 2024) [abs] | hardware UF decoding, per-round latency | no symmetry-indexed weight memory found in the abstract; full text to be checked |
| Das, Locharla, Jones, Chong, Martonosi, Tannu, *LILLIPUT*, arXiv:2108.06569 [snip] | lookup-table decoder; compression keeps only probable entries (107x) | compression by probability, not by symmetry |
| Barber et al., Collision Clustering (Riverlane) [snip] | low-memory UF variant in hardware | no symmetry compression reported in what was seen |

## 5. Testing and verification of decoders

| Work | What it exploits | Delta |
|---|---|---|
| Shen, Zhong, *Capability-gated conformance testing of quantum error-correction decoder libraries*, arXiv:2609.07035 (submitted to ACM TQC) [full: abstract, Sections 7 and 9 excerpts] | oracle-free metamorphic tests: two presentations of one instance (relabellings) must give equal minimum weight; a **"strict equivariance" relation that requires an automorphism preserving the full weighted model including observable masks**; they report such automorphisms exist at circuit level on the rotated surface code (dispatched 4,320 times, never fired); mutation study of 13 interface operations on software decoders | the closest work to H5. Differences: they test software libraries (PyMatching, fusion-blossom, BP-OSD), we test an HLS-level hardware decoder model with hardware-style faults (weight bit flips, address swaps, boundary off-by-one); our relation uses the space-time reflection, which maps the observable to an equivalent one rather than preserving the mask, so it is not their strict relation; we compare detection per test vector against differential testing |
| Maan, Paler, *Testing the accuracy of surface code decoders*, arXiv:2311.12503 [snip] | exhaustive comparison against lookup tables up to d = 7 | oracle based, not metamorphic |
| Abreu et al., *Metamorphic testing of oracle quantum programs* (Q-SE 2022) [snip] | metamorphic testing of quantum programs | not decoders |

## 6. DEM construction and merging

| Work | Relevance |
|---|---|
| Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021) | Stim merges error mechanisms with identical detector and observable footprints into one DEM line; `explain_detector_error_model_errors` lists the circuit faults behind each line. Twin contraction of the fault graph should reproduce this merge; H1 tests it rather than assuming it |
| *Composing detector error models*, arXiv:2609.13087 [snip] | DEM composition; to be checked for overlap with H1 |
| *Reconstruction of detector error model for quantum error correction*, arXiv:2606.16288 [snip] | estimation of DEM hypergraphs from data; not symmetry |
| Prakash, *A short proof of twin-quotient decomposition and orbit lifting*, Preprints.org doi:10.20944/preprints202609.1249.v1 [full] | the theorem used here (Aut(G) is K ⋊ A, exact orbit lifting); GraphCrunch implements it |
| McKay, Piperno, *Practical graph isomorphism II*, J. Symb. Comput. 60 (2014) 94 to 112 | nauty |

## Searches with no direct hit

- graph automorphisms of surface code decoding graphs computed with nauty or by twin contraction:
  no hit other than AutDEC (Bliss on DEM Tanner graphs of qLDPC codes) and Shen and Zhong (weighted
  automorphisms used as a test relation);
- twin vertices or modular decomposition applied to detector error models: no hit;
- symmetry-indexed weight memory in FPGA or ASIC matching or Union-Find decoders: no hit;
- rotation or reflection data augmentation for circuit-level surface code neural decoders: no hit.

A negative search is weak evidence. The paper will say "we did not find" and cite the searches, and
will not claim priority.

## Proposed statement of what this project adds (for your approval or edit)

Automorphisms of circuit-level decoding problems have been computed before, with Bliss on the
detector error model Tanner graphs of bivariate bicycle codes, to drive automorphism ensemble
belief propagation (AutDEC); weighted automorphisms of rotated surface code models have been used
as a metamorphic relation for software decoder libraries (Shen and Zhong); and time translation has
been used to tie decoder priors during calibration (Sivak et al.). This project differs in four
ways. (i) It derives the decoding graph and its group exactly from the circuit fault graph by twin
contraction with a proved lift, so that the merged detector error model, the twin group K and the
residual group of the decoding graph come out of one computation, and it measures, across distance,
basis and noise model, which group elements survive the noise weights and in which rounds they are
broken. (ii) It tests whether the computed space-time point reflection, which is not a lattice
translation and is not exact in the tensor layout of NVIDIA's pre-decoder, improves a 3D
convolutional pre-decoder when imposed as augmentation or as weight tying, including when evaluated
beyond the training distance. (iii) It measures what orbit-indexed weight memory saves in an HLS
decoder memory, at HLS and post-route level, against a flat table. (iv) It uses the reflection as
an oracle-free invariant for an HLS-level decoder model under hardware-style fault injection, and
the disagreement between a shot and its reflected twin as an escalation signal next to a confidence
gate, and compares both against existing baselines (differential testing, ensemble flagging). Each
of (ii) to (iv) may give a negative result, which will be reported as such.

## Consequences for the hypotheses (no change made; for your decision)

- **H5:** Shen and Zhong already show that, on software libraries, weight-preserving relabellings
  catch real defects and that strict equivariance never fired. H5 stays meaningful for hardware
  faults but must be framed against their result, and should include their relabelling relation as
  a fourth test policy so the comparison is fair.
- **H6:** harmonization already uses ensemble disagreement to flag risky shots. H6 should compare
  the σ-twin bit with a random-perturbation twin of equal cost, otherwise an improvement could come
  from "any second opinion" rather than from the symmetry.
- **H3:** the residual group of the decoding graph is expected to have order 2 (the regression
  table), which bounds the gain from augmentation to at most a factor two in effective data. A null
  result is the most likely outcome and the experiment design (fixed budget, paired seeds) should be
  powered for a small effect.
