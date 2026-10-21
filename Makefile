# automorphism-aware-decoder
# Third-party trees live beside the repository (third_party/README.md).
PY        ?= ../.venv/bin/python
THIRD     ?= ../third_party
NAUTYDIR  ?= $(THIRD)/GraphCrunch/nauty
export PYTHONPATH := src

.PHONY: tools test lint phase1 figures clean

tools: build/nauty_aut

build/nauty_aut: tools/nauty_aut.c
	mkdir -p build
	gcc -O3 -Wall -D_GNU_SOURCE -I$(NAUTYDIR) -o $@ $< $(NAUTYDIR)/nauty.a -lm

test: tools
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests bench

# Phase 1: reproduction of the regression table, the symmetry sweep, the fault-graph group check,
# exports and figures (about 45 minutes on the development host, most of it plain nauty at d = 5, 7)
phase1: tools
	$(PY) bench/phase1_reproduction.py
	$(PY) bench/symmetry_sweep.py
	$(PY) bench/faultgraph_group.py
	$(PY) bench/export_graphs.py
	$(PY) bench/plot_phase1.py

figures:
	$(PY) bench/plot_phase1.py

clean:
	rm -rf build .pytest_cache
