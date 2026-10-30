"""Ring J=2.5x on the 616 campaign placements (chip_campaign.py --problems ring:2.5) against the prediction
pre-registered in commit 6a0a6dc (results/prediction_ring25x.json / .csv), and its consequence for the readout
ambiguity of the ring 1.75x (tau_sensitivity step 0).

  1. hardware 1-P_GS per t_a: pooled over all placements and takes (precise), chip median, take-to-take spread
  2. the prediction test: the t_a dependence of the floor (31.5-242 ns). Fixed budget B: the thermal part is
     <= 2e-5, so the hardware should be flat at its readout level; constant tau: it should rise ~80x from 58 to 242 ns
  3. the new readout bound (the ring 2.5x floor on the same qubits) and the ring 1.75x refit with it

Usage:  python ring25_check.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import chip_campaign_analysis as cca
import tau_sensitivity as ts
import tau_sensitivity_analysis as tsa

RES = cca.RES
LOG = RES / "chip_campaign_ring25x_log.csv"
FLOOR_FROM_NS = 31.0


def hardware():
    d = pd.read_csv(LOG, dtype={"take_id": str, "j_sign": str})
    d["t_anneal_ns"] = d.t_anneal_ns.round(3)
    d["fail"] = (1 - d.p_gs) * d.num_reads
    rows = []
    for (sg, ta), g in d.groupby(["j_sign", "t_anneal_ns"]):
        per_take = g.groupby("take_id").apply(lambda x: x.fail.sum() / x.num_reads.sum())
        pooled = g.fail.sum() / g.num_reads.sum()
        rows.append(dict(sign=sg, t_a_ns=ta, n_takes=g.take_id.nunique(), om_pooled=pooled,
                         om_pooled_sem=per_take.std() / np.sqrt(len(per_take)) if len(per_take) > 1 else np.nan,
                         om_tile_median=float((1 - g.groupby("tile").p_gs.mean()).median()),
                         failures_total=int(round(g.fail.sum())), reads_total=int(g.num_reads.sum())))
    return d, pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pred = json.loads((RES / "prediction_ring25x.json").read_text())
    P = pd.read_csv(RES / "prediction_ring25x.csv").pivot_table(index="t_a_ns", columns="model", values="p_gs")
    om_pred = 1 - P
    d, H = hardware()
    print(f"ring 2.5x: {d.take_id.nunique()} takes, {d.tile.nunique()} placements, {len(H)} (sign, t_a) points")
    print(H.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    print("\n=== prediction test on the floor (t_a >= 31.5 ns) ===")
    rows = []
    for sg, h in H.groupby("sign"):
        h = h.set_index("t_a_ns").sort_index()
        fl = h[h.index >= FLOOR_FROM_NS]
        t, om = fl.index.values, fl.om_pooled.values
        slope = np.polyfit(np.log(t), np.log(om), 1)[0]
        eps_hat = float(np.mean(om))                         # readout = the flat level, if flat
        for m in om_pred.columns:
            mod = om_pred[m].reindex(t).values
            for eps_mode, eps in (("eps=fitted level", None), ("eps=0", 0.0)):
                # with eps free (0..max hw): best constant offset in log space
                grid = np.linspace(0, om.max(), 2001) if eps is None else np.array([0.0])
                best = min((float(np.sqrt(np.mean(np.log10((1 - (1 - mod) * (1 - e)) / om) ** 2))), e) for e in grid)
                rows.append(dict(sign=sg, model=m, eps_mode=eps_mode, eps=best[1], rms_log=best[0], factor=10 ** best[0],
                                 hw_ratio_last_over_58=float(h.om_pooled.loc[241.817] / h.om_pooled.loc[58.189]),
                                 model_ratio_last_over_58=float(om_pred[m].loc[241.817] / om_pred[m].loc[58.189]),
                                 hw_loglog_slope=slope))
    T = pd.DataFrame(rows)
    T.to_csv(RES / "ring25_prediction_test.csv", index=False)
    print(T.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    print("\n=== new readout bound and the ring 1.75x refit ===")
    out = tsa.compute()
    taus, ta, om, omf = out["grids"][("ring", 1.75)]
    fl = slice(len(ta) - tsa.N_FLOOR, len(ta))
    rows = []
    for sg in ("-", "+"):
        h = H[(H.sign == sg) & (H.t_a_ns >= FLOOR_FROM_NS)]
        # readout is t_a-independent, so it cannot exceed the SMALLEST floor value measured on these qubits
        new_bound = float(h.om_pooled.min())
        old_bound = float(out["eps_bound"][out["eps_bound"].j_sign == sg].eps_max.median())
        ss = cca._sel(out["s"], "ring", 1.75, sg).set_index("t_anneal_ns").reindex(ta)
        hw = (1 - ss.p_median).values[fl]
        s_c = ts.problem_info("ring", 1.75)[0]
        for label, emax in (("old bound", old_bound), ("new bound", new_bound), ("no readout", 0.0)):
            tb, eb, l, lo, hi = tsa.fit(omf[:, fl], hw, emax)
            # fixed budget B (tau = s_c t_a / B per anneal), interpolated on the fine tau grid as in tau_budget_checks
            best = None
            for B in np.exp(np.linspace(np.log(0.5), np.log(40), 400)):
                tau_t = s_c * ta[fl] / B
                if tau_t.min() < taus.min() or tau_t.max() > taus.max():
                    continue
                om_t = np.array([np.exp(np.interp(np.log(t_), np.log(taus), np.log(om[:, k])))
                                 for t_, k in zip(tau_t, range(len(ta) - tsa.N_FLOOR, len(ta)))])
                for e in np.linspace(0, emax, 41):
                    lb = float(np.sqrt(np.mean(np.log10((1 - (1 - om_t) * (1 - e)) / hw) ** 2)))
                    if best is None or lb < best[0]:
                        best = (lb, B, e)
            rows.append(dict(sign=sg, readout=label, eps_max=emax, tau=tb, eps=eb, factor_const_tau=10 ** l, at_edge=lo or hi,
                             B=best[1], eps_B=best[2], factor_fixed_B=10 ** best[0]))
    R = pd.DataFrame(rows)
    R.to_csv(RES / "ring25_ring175_refit.csv", index=False)
    print(R.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nprediction metadata:", json.dumps(pred["B_predictions"]), "| created", pred["created"])


if __name__ == "__main__":
    main()
