"""Candidate group elements built from coordinates, and orbit tables for export.

``reflection_candidates`` builds, for each stabilizer sector, the space-time point reflection of that
sector's detectors through the centre of their coordinate box, leaving the other sector and the
boundary fixed. Whether a candidate is an automorphism is then decided by
``symmetry.is_automorphism``; nothing here assumes it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autsym.circuits import X_STAB, Z_STAB, DetectorInfo
from qec.dem import DecodingGraph


def reflection(det: DetectorInfo, n_nodes: int, sector: int) -> np.ndarray | None:
    """Point reflection of one sector, or None if some image is not a detector of that sector."""
    m = det.stab == sector
    if not m.any():
        return None
    cx = int(det.x[m].min() + det.x[m].max())
    cy = int(det.y[m].min() + det.y[m].max())
    ct = int(det.t[m].min() + det.t[m].max())
    where = {(int(x), int(y), int(t)): i for i, (x, y, t) in enumerate(zip(det.x, det.y, det.t)) if m[i]}
    perm = np.arange(n_nodes, dtype=np.int64)
    for i in np.flatnonzero(m):
        j = where.get((cx - int(det.x[i]), cy - int(det.y[i]), ct - int(det.t[i])))
        if j is None:
            return None
        perm[i] = j
    return perm


def reflection_candidates(det: DetectorInfo, n_nodes: int) -> dict[str, np.ndarray]:
    out = {}
    for name, s in (("X", X_STAB), ("Z", Z_STAB)):
        p = reflection(det, n_nodes, s)
        if p is not None:
            out[f"reflect_{name}"] = p
    if "reflect_X" in out and "reflect_Z" in out:
        out["reflect_both"] = out["reflect_X"][out["reflect_Z"]]
    return out


def orbit_table(g: DecodingGraph, cls: np.ndarray) -> dict:
    """Class table for an orbit-indexed weight memory: per class its representative edge (lowest
    index), member count and weight; plus the class of every edge."""
    reps: dict[int, int] = {}
    count: dict[int, int] = {}
    for i, c in enumerate(cls.tolist()):
        reps.setdefault(c, i)
        count[c] = count.get(c, 0) + 1
    classes = [{"id": c, "representative_edge": reps[c], "members": count[c],
                "weight": float(g.weight[reps[c]])} for c in sorted(reps)]
    return {"n_edges": g.n_edges, "n_classes": len(classes), "classes": classes, "edge_class": cls.tolist()}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
