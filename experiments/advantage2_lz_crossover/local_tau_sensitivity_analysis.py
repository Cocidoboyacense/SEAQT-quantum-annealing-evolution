"""Analysis for local_tau_sensitivity.py: the steps of 16.10 applied to the local SEAQT dissipator at T = 70 mK,
plus the elasticity in T (at tau = 300 ns) and a fixed-budget-like reparametrization (tau proportional to t_a).
Invalid runs (P_gs > 1, or failed integrations) are dropped; every remaining run's min_eig is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import chip_campaign_analysis as cca
import local_tau_sensitivity as lts
import tau_sensitivity as ts
import tau_sensitivity_analysis as tsa

RES = ts.RESULTS_DIR
PLOTS = RES / "plots"


def valid(d: pd.DataFrame) -> pd.DataFrame:
    return d[d.p_gs.notna() & (d.p_gs <= 1.0)]


def runs_at(temp=lts.T_MAIN):
    d = valid(lts.all_runs())
    return d[np.isclose(d.T_mK, temp)][["topology", "j_multiple", "tau_ns", "t_a_ns", "p_gs", "min_eig"]]


def compute():
    tsa.TAU_FINE = np.exp(np.linspace(np.log(min(lts.TAU_GRID)), np.log(max(lts.TAU_GRID)), 400))
    r = runs_at()
    out = tsa.compute(r)
    out["summary"] = tsa.summarize(out)
    out["min_eig"] = float(r.min_eig.min())
    allr = lts.all_runs()
    out["n_invalid"] = int(len(allr) - len(valid(allr)))
    out["temp"] = temp_elasticity()
    out["budget"] = budget_fit(out)
    return out


def temp_elasticity():
    """d ln(1-P) / d ln T at T = 70 mK (central difference 60/80 mK), tau = 300 ns, per problem and t_a."""
    d = valid(lts.all_runs())
    d = d[np.isclose(d.tau_ns, lts.TAU_MAIN)]
    rows = []
    for (topo, jm), g in d.groupby(["topology", "j_multiple"]):
        P = g.pivot_table(index="T_mK", columns="t_a_ns", values="p_gs")
        if not {60.0, 70.0, 80.0} <= set(P.index):
            continue
        for ta in P.columns:
            om = 1 - P[ta]
            if om[[60.0, 80.0]].isna().any() or (om[[60.0, 80.0]] <= 0).any():
                continue
            rows.append(dict(problem=f"{topo} J={jm:g}x", t_a_ns=ta, om_70=float(om[70.0]),
                             elasticity_T=float(np.log(om[80.0] / om[60.0]) / np.log(80.0 / 60.0))))
    return pd.DataFrame(rows)


def budget_fit(out):
    """tau = t_a / b (a fixed relaxation budget per anneal, b = t_a/tau; the local bath has no switch), interpolated on
    the tau grid, against the constant-tau fit, both with the bounded readout term; sign by sign, chip median."""
    rows = []
    eb = out["eps_bound"]
    for topo, jm in ts.PROBLEMS:
        taus, ta, om, _ = out["grids"][(topo, jm)]
        fl = slice(len(ta) - tsa.N_FLOOR, len(ta))
        for sg in ("-", "+"):
            ss = cca._sel(out["s"], topo, jm, sg).set_index("t_anneal_ns").reindex(ta)
            hw = np.clip((1 - ss.p_median).values[fl], 1e-9, None)
            emax = float(eb[eb.j_sign == sg].eps_max.median())
            best = None
            for b in np.exp(np.linspace(np.log(0.005), np.log(5.0), 500)):
                tau_t = ta[fl] / b
                if tau_t.min() < taus.min() or tau_t.max() > taus.max():
                    continue
                om_t = np.array([np.exp(np.interp(np.log(t_), np.log(taus), np.log(om[:, k])))
                                 for t_, k in zip(tau_t, range(len(ta) - tsa.N_FLOOR, len(ta)))])
                for e in np.linspace(0, emax, 21):
                    l_ = float(np.sqrt(np.mean(np.log10((1 - (1 - om_t) * (1 - e)) / hw) ** 2)))
                    if best is None or l_ < best[0]:
                        best = (l_, b, e, tau_t)
            const = out["step0"].query("problem == @tsa.label(@topo, @jm) and sign == @sg").step_rms_log.iloc[0]
            rows.append(dict(problem=tsa.label(topo, jm), sign=sg, factor_const_tau=10 ** const,
                             b=np.nan if best is None else best[1], factor_budget=np.nan if best is None else 10 ** best[0],
                             tau_range=None if best is None else f"{best[3].min():.0f}-{best[3].max():.0f}"))
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd.set_option("display.width", 250)
    out = compute()
    fmt = lambda x: f"{x:.4g}"
    print(f"local dissipator, T = 70 mK: {len(runs_at())} valid anneals; invalid (P>1 or failed) over all T: {out['n_invalid']}; "
          f"worst min_eig at 70 mK: {out['min_eig']:+.2e}")
    print("\n=== step 0 ===")
    print(out["step0"][["problem", "sign", "eps_bound", "closed_rms_log", "step_tau", "step_eps", "step_rms_log"]]
          .to_string(index=False, float_format=fmt))
    s1 = out["step1"]
    print("\n=== step 1 (sign -) ===")
    print(s1[s1.sign == "-"][["problem", "t_a_ns", "floor", "om_hw", "elasticity", "drift_rel", "tau_resolution_rel",
                              "best_tau_this_ta"]].to_string(index=False, float_format=fmt))
    print("\n=== steps 2-3 ===")
    print(out["summary"][["problem", "sign", "tau_takes_median", "tau_takes_p10", "tau_takes_p90", "tau_tiles_median",
                          "tau_tiles_p10", "tau_tiles_p90", "tiles_at_edge"]].to_string(index=False, float_format=fmt))
    print("\n=== elasticity in T (tau = 300 ns) ===")
    print(out["temp"].to_string(index=False, float_format=fmt))
    print("\n=== tau proportional to t_a vs constant tau (floor) ===")
    print(out["budget"].to_string(index=False, float_format=fmt))
    for k in ("step0", "step1", "takes", "tiles", "summary", "temp", "budget"):
        out[k].to_csv(RES / f"local_tau_sensitivity_{k}.csv", index=False)
    for name, f in (("curves", tsa.fig_curves), ("elasticity", tsa.fig_elasticity), ("distributions", tsa.fig_distributions)):
        fig = f(out)
        if fig._suptitle is not None:
            fig._suptitle.set_text(fig._suptitle.get_text().replace("Step-at-$s_c$ global bath", "Local SEAQT, 70 mK"))
        fig.savefig(PLOTS / f"local_tau_sensitivity_{name}.png", dpi=110)
        plt.close(fig)
    return out


if __name__ == "__main__":
    main()
