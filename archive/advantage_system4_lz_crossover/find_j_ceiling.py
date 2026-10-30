"""Find the max self-consistently achievable coupling energy at the
critical point s_c on Advantage_system4's real schedule (extended_j_range
confirmed [-2.0, 1.0], same asymmetric pattern as Advantage2_system1) --
same methodology used to design Advantage2's J_MULTIPLES. Prints candidate
multiples of k_B T/h (T=15.4 mK) and verifies each gives a genuinely
distinct, unclipped s_c/J pair before committing them to config.py.
"""
import numpy as np
import config

schedule = config.load_system4_schedule()
EXTENDED_J_RANGE = (-2.0, 1.0)  # confirmed via DWaveSampler(solver='Advantage_system4').properties

Gamma = lambda s: schedule.A(s) / 2.0 * 1000.0
Jcal = lambda s: schedule.B(s) / 2.0 * 1000.0

print(f"K_BT_OVER_H_MHZ (T={config.T_OPERATING_MK} mK) = {config.K_BT_OVER_H_MHZ:.4f} MHz")
print(f"Gamma(0)={Gamma(1e-6):.1f} MHz, Gamma(1)={Gamma(1-1e-6):.1f} MHz")
print(f"Jcal(0)={Jcal(1e-6):.1f} MHz, Jcal(1)={Jcal(1-1e-6):.1f} MHz\n")

# Binary-search the largest target_mhz that does NOT require clipping
# (i.e. j_abs = target_mhz/Jcal(s_c) <= max(|extended_j_range|)) --
# same ceiling-finding idea as the Advantage2 module's README analysis.
lo_j, hi_j = EXTENDED_J_RANGE
max_mag = max(abs(lo_j), hi_j)

def j_abs_for_target(target_mhz):
    from scipy.optimize import brentq
    s_c = brentq(lambda s: Gamma(s) - target_mhz, 1e-6, 1 - 1e-6)
    return s_c, target_mhz / Jcal(s_c)

lo_t, hi_t = 1.0, Gamma(1e-6) - 1.0  # search range in MHz
for _ in range(60):
    mid = 0.5 * (lo_t + hi_t)
    s_c, j_abs = j_abs_for_target(mid)
    if j_abs <= max_mag:
        lo_t = mid
    else:
        hi_t = mid
ceiling_mhz, s_c_ceiling = lo_t, j_abs_for_target(lo_t)[0]
ceiling_mult = ceiling_mhz / config.K_BT_OVER_H_MHZ
print(f"Max unclipped target: {ceiling_mhz:.1f} MHz (s_c={s_c_ceiling:.4f}) "
      f"= {ceiling_mult:.3f}x k_B T/h\n")

# Candidate multiples spread comfortably under the ceiling, same spirit as
# Advantage2's [1.0, 1.75, 2.5, 3.25, 4.0] (kept ~4x under a ~4.1x ceiling).
candidates = list(np.linspace(1.0, ceiling_mult * 0.95, 5))
print(f"Candidate J_MULTIPLES (95% of ceiling): {[round(c,3) for c in candidates]}\n")

print(f"{'multiple':>9} {'target_mhz':>11} {'s_c':>8} {'j_abs':>8} {'clipped?':>9}")
for m in candidates:
    target = m * config.K_BT_OVER_H_MHZ
    s_c, j_abs = j_abs_for_target(target)
    clipped = j_abs > max_mag
    print(f"{m:9.3f} {target:11.1f} {s_c:8.4f} {j_abs:8.4f} {str(clipped):>9}")
