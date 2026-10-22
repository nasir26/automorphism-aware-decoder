# Limitations

## Phase 0

- No GPU on this host; every training run in later phases is CPU only unless moved.
- Prior-art survey: entries marked [snip] in `docs/PRIOR_ART.md` were seen only as search results;
  they are not cited until checked. Negative searches are weak evidence and support only "we did not
  find", never a priority claim.
- `make test` has nothing to run yet: the venv packages are not installed (waiting for approval).

## Phase 1

- H1's footprints come from Stim's `explain_detector_error_model_errors`, and its reference is
  Stim's own DEM. The test therefore shows that twin contraction plus lifting reproduces Stim's
  merged DEM exactly and preserves the automorphism group; it is not an independent check of Stim's
  error propagation.
- Weight equality is decided at 9 significant digits. At exact binary64 equality more pairs differ
  (683 instead of 469 at d = 13, basis X), from rounding in the probability to weight conversion.
  Every group statement with weights is made at 9 digits and says so.
- The regression table's "78 distinct exact weights" was not reproduced (D-1 in DECISIONS.md).
- Plain nauty on the uncontracted fault graph was run up to d = 5 in every configuration and at
  d = 7 for NVIDIA basis X only; beyond that only the contracted solve was run. The d = 13
  uncontracted run fails, as in the table.
- Two noise families only (NVIDIA single p, Stim uniform depolarizing), one p (0.003). SI1000 is not
  swept.
- d = 3 is a special case (bare group of order 32); statements about σ are made for d >= 5.

## Phase 2

- Every Phase 2 result uses PyMatching 2.4.0, which rounds weights to integers internally; "minimum
  weight" means minimum under that rounding.
- The σ-twin sweep covers NVIDIA's circuit, basis X only, d = 5 to 13, p = 0.002 to 0.008 at one
  noise model. d = 13, p = 0.002 is inconclusive (fewer than 100 disagreeing shots in 76.8 M).
- Automorphism ensemble decoding was evaluated with a group of order 2 and an exact matcher, where it
  cannot gain by construction; it says nothing about ensembles of approximate decoders, which is
  where AutDEC applies it.

## Phase 3

- One budget only: width 64, depth 2, 20 epochs of 2,000,000 shots at p = 0.006, trained at d = 5,
  CPU. A larger model or budget could change the comparison with PyMatching; the null is stated for
  this budget.
- One circuit family (Stim rotated memory, uniform depolarizing), one basis (Z), one test p (0.003).
- Every model is applied with a single threshold chosen at d = 5. At larger d the ratio to PyMatching
  alone grows (about 3 at d = 13), so the pre-decoder's local errors are not removed by retraining or
  rethresholding per d; that was not tried.
- The equivariant arm's syndrome density reduction falls with d (0.50 at d = 5 to 0.44 at d = 13),
  while the other arms stay at 0.50 to 0.53; it applies fewer (0, -1, 1) space edges at d = 13
  (4.0 per shot against 5.5 to 5.7). The cause is not established.
- d = 11 and 13 have 6 to 42 failures per model; their paired comparisons are weak.
- FP32 equivariance deviation (up to 1.1e-5) is above the prompt's 1e-6 target; the FP64 value
  (1.8e-14) shows it is rounding, not construction.

## Phase 4

- QAT starts from the Phase 3 checkpoints, which do not beat PyMatching alone; Phase 4 measures how
  much quantization costs, not a useful decoder.
- Evaluated at d = 5 and 9 only; the augmented and TTS variants are not quantized.
- The head accumulator reaches 2.9e10 in magnitude at INT8 (35 bits plus sign); a hardware datapath
  needs at least that width, or a smaller head multiplier.
- The golden model is software (T4). Nothing here has been synthesised or run on a card.
