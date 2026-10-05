#!/usr/bin/env python3
"""Feasibility checks for the continuous byte-mass approximation (not a codec theorem)."""
from __future__ import annotations
import json

def slope_interval(density: list[float], minimum_bytes: list[float], raw_bytes: list[float]) -> dict:
    if not density or len(density)!=len(minimum_bytes) or len(density)!=len(raw_bytes):raise ValueError('length mismatch')
    if any(r<=0 or lo<=0 or hi<lo for r,lo,hi in zip(density,minimum_bytes,raw_bytes)):raise ValueError('invalid positive density/byte bounds')
    lower=max(r*lo for r,lo in zip(density,minimum_bytes));upper=min(r*hi for r,hi in zip(density,raw_bytes))
    return {'feasible_in_continuous_relaxation':lower<=upper,'slope_lower':lower,'slope_upper':upper}

if __name__=='__main__':
    print(json.dumps({'twofold_density_key_only':slope_interval([1,2],[1,1],[8,8]),
                      'tenfold_density_key_only':slope_interval([1,10],[1,1],[8,8]),
                      'tenfold_density_with_8_byte_values':slope_interval([1,10],[9,9],[16,16])},indent=2))
