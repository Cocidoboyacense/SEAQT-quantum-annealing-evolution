"""Coherent-window fit (a_empirical) vs. closed-system LZ theory (a_theory,
King et al. 2022 Eq. 8) for Advantage_system4's refinement data -- same
methodology as experiments/advantage2_lz_crossover's section-16 analysis
and stage1_full.py's a_theory_for.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

schedule = config.load_system4_schedule()
Gamma = lambda s: schedule.A(s) / 2.0
Jcal = lambda s: schedule.B(s) / 2.0
eps = 1e-5


def a_theory_for(s_c):
    Gp = (Gamma(s_c + eps) - Gamma(s_c - eps)) / (2 * eps)
    Jp = (Jcal(s_c + eps) - Jcal(s_c - eps)) / (2 * eps)
    b = Gamma(s_c) / (Jp / Jcal(s_c) - Gp / Gamma(s_c))
    return 2 * np.pi**3 * b / config.N_QUBITS**2


def first_local_min(y):
    for i in range(len(y) - 1):
        if y[i + 1] >= y[i]:
            return i
    return len(y) - 1


df = pd.read_csv("results/refinement.csv")
j_multiples = sorted(df.j_multiple.unique())
MIN_FIT_PTS = 3
FLOOR = 0.5 / 8000

fig, axes = plt.subplots(2, len(j_multiples), figsize=(4 * len(j_multiples), 7), sharey=False)
summary = []
for row, topo in enumerate(["ring", "chain"]):
    for col, jm in enumerate(j_multiples):
        ax = axes[row, col]
        s = df[(df.topology == topo) & (df.j_multiple == jm)].sort_values("t_anneal_ns").reset_index(drop=True)
        t_all = s.t_anneal_ns.values
        s_c = s.s_c.iloc[0]
        a_th = a_theory_for(s_c)
        one_minus = np.clip(1 - s.p_ground.values, FLOOR, None)
        peak_idx = first_local_min(one_minus)
        t_pre, y_pre = t_all[:peak_idx + 1], one_minus[:peak_idx + 1]

        ax.semilogy(t_all, one_minus, "o", color="tab:gray", alpha=0.35, ms=4)
        ax.semilogy(t_pre, y_pre, "o", color="tab:blue" if topo == "ring" else "tab:red", ms=5)
        t_th = np.linspace(0, t_all.max() * 1.05, 100)
        ax.semilogy(t_th, np.exp(-a_th * t_th), "--", color="gray", lw=1)

        n_fit = len(t_pre)
        if n_fit >= MIN_FIT_PTS:
            slope, intercept = np.polyfit(t_pre, np.log(y_pre), 1)
            a_emp = -slope
            ax.semilogy(t_th, np.exp(intercept) * np.exp(-a_emp * t_th), "-", color="black", lw=1)
            ratio = a_emp / a_th
            summary.append((topo, jm, s_c, n_fit, a_th, a_emp, ratio))
            title = f"{topo}, J={jm:g}x\na_emp/a_th={ratio:.2f} (n={n_fit})"
        else:
            summary.append((topo, jm, s_c, n_fit, a_th, None, None))
            title = f"{topo}, J={jm:g}x\nonly {n_fit} pt(s)"
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("$t_a$ (ns)", fontsize=8)
        if col == 0:
            ax.set_ylabel("$1-P_{GS}$")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

fig.suptitle("Advantage_system4: refined coherent-window fit vs. closed-system LZ theory")
fig.tight_layout()
fig.savefig("results/refinement_fit.png", dpi=140)
print("saved results/refinement_fit.png\n")

print(f"{'topo':6s} {'J':>5s} {'s_c':>7s} {'n_fit':>6s} {'a_theory':>10s} {'a_empirical':>12s} {'ratio':>7s}")
for topo, jm, s_c, n_fit, a_th, a_emp, ratio in summary:
    if a_emp is not None:
        print(f"{topo:6s} {jm:5.2f} {s_c:7.4f} {n_fit:6d} {a_th:10.5f} {a_emp:12.5f} {ratio:7.3f}")
    else:
        print(f"{topo:6s} {jm:5.2f} {s_c:7.4f} {n_fit:6d} {a_th:10.5f} {'--':>12s} {'--':>7s}")

ratios = [r[6] for r in summary if r[6] is not None]
print(f"\nn={len(ratios)}, mean={np.mean(ratios):.3f}, range=[{min(ratios):.3f}, {max(ratios):.3f}]")
print(f"All >1: {all(r>1 for r in ratios)}")

pd.DataFrame(summary, columns=["topology","j_multiple","s_c","n_fit","a_theory","a_empirical","ratio"]) \
    .to_csv("results/refinement_analysis_summary.csv", index=False)
