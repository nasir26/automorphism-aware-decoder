# Claims ledger

Every number in the README, figures or paper has a row here and a file under `results/`.

**Tiers:** `T1-HW` measured on the U55C via XRT from the C++ host only; `T2-POSTROUTE`
post-place-and-route reports; `T3-HLS` Vitis HLS csynth estimates; `T4-SIM` software simulation or
analytic model. `PROJECTED` may appear here but never in the abstract, title, conclusions or a
results figure.

**Exactness:** `exact` means integer, rational or bit-identical comparison. Weight equality is
decided at a stated number of significant digits, recorded in the row.

| Id | Claim | Value | Tier | Exact or digits | Evidence | Status |
|---|---|---|---|---|---|---|
| E-001 | GraphCrunch brute-force self test on this host | 1,258 graphs, 0 failures | T4-SIM | exact | `results/env.json` | supported |
