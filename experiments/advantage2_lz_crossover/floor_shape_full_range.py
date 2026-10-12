"""Empirical shape of the hardware's 1-P_GS beyond the coherent regime, over the full range (no simulation):
the exploratory campaign (1000 reads, one placement, 5 ns - ~50 us) and the chip campaign (8000 reads x 24 takes,
616 placements, 5-145 ns). Fits two simple forms on regime II (from the peak of P_GS to its minimum):
    linear   1-P = a + b t_a
    power    1-P = c t_a^h
and checks whether a form fitted on 38-145 ns (the window where the fixed budget B was found) extrapolates to the
microsecond minimum.

Usage:  python floor_shape_full_range.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import chip_campaign_analysis as cca

RES = cca.RES
CASES = [("ring", 1.0, "-"), ("ring", 1.0, "+"), ("chain", 1.0, "-"), ("chain", 1.0, "+"), ("ring", 1.75, "-"), ("ring", 1.75, "+")]


def regime_two(g: pd.DataFrame) -> pd.DataFrame:
    """Points from the P_GS peak to the following minimum (inclusive)."""
    g = g.sort_values("t_anneal_ns").reset_index(drop=True)
    i = int(np.argmax(g.p_ground.values))
    k = i + int(np.argmin(g.p_ground.values[i:]))
    return g.iloc[i:k + 1]


def fits(t, om):
    lin = np.polyfit(t, om, 1)                                   # [b, a]
    pw = np.polyfit(np.log(t), np.log(om), 1)                     # [h, ln c]
    rms = lambda pred: float(np.sqrt(np.mean(np.log10(np.clip(pred, 1e-9, None) / om) ** 2)))
    return dict(lin_a=lin[1], lin_b_per_ns=lin[0], lin_rms_log=rms(np.polyval(lin, t)),
                pow_h=pw[0], pow_c=float(np.exp(pw[1])), pow_rms_log=rms(np.exp(np.polyval(pw, np.log(t)))))


def main():
    ex = pd.read_csv(RES / "exploratory.csv", dtype={"j_sign": str})
    d, s, M, A = cca.compute(verbose=False)
    rows = []
    for topo, jm, sg in CASES:
        g = ex[(ex.topology == topo) & np.isclose(ex.j_multiple, jm) & (ex.j_sign == sg)]
        r2 = regime_two(g)
        t, om = r2.t_anneal_ns.values, 1 - r2.p_ground.values
        f_ex = fits(t, om)
        # chip campaign floor window (last 4 t_a), chip median
        ss = cca._sel(s, topo, jm, sg).sort_values("t_anneal_ns").tail(4)
        tc, oc = ss.t_anneal_ns.values, (1 - ss.p_median).values
        f_c = fits(tc, oc)
        t_min, om_min = float(t[-1]), float(om[-1])
        rows.append(dict(problem=f"{topo} J={jm:g}x", sign=sg,
                         expl_peak_ns=float(t[0]), expl_om_peak=float(om[0]), expl_min_ns=t_min, expl_om_min=om_min,
                         expl_n=len(t), expl_lin_b_pct_per_us=100 * f_ex["lin_b_per_ns"] * 1000, expl_lin_rms_log=f_ex["lin_rms_log"],
                         expl_pow_h=f_ex["pow_h"], expl_pow_rms_log=f_ex["pow_rms_log"],
                         camp_lin_b_pct_per_us=100 * f_c["lin_b_per_ns"] * 1000, camp_pow_h=f_c["pow_h"],
                         camp_lin_extrap_at_min=f_c["lin_a"] + f_c["lin_b_per_ns"] * t_min,
                         camp_pow_extrap_at_min=f_c["pow_c"] * t_min ** f_c["pow_h"]))
    T = pd.DataFrame(rows)
    T.to_csv(RES / "floor_shape_full_range.csv", index=False)
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return T


if __name__ == "__main__":
    main()
