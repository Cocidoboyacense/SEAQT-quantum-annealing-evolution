"""Minimal end-to-end smoke test against REAL hardware, before committing
to the full exploratory sweep (215 combos, ~2.58M reads). Exercises the
new code paths this session added, at low cost (3 points, 100 reads
each -- 300 total reads):

  1. ring, j_sign=+1        -- abstract-topology embedding (minorminer,
                                shared with chain), explicit sign threading
  2. a plain native topology, j_sign=None (auto) -- native fixed-qubit
                                embedding (no minorminer), auto sign
  3. native_e6_c26_r0, j_sign=+1 -- the near-zero-true-gap frustrated case
                                found this session; also a real first look
                                at its actual P_gs

get_sampler() itself (called once, building embeddings for all 35
topologies) is also exercised here for the first time against live
hardware in this session.

Usage:
    python smoke_test_hardware.py
"""
from submit_hardware import get_sampler, run_point
import config

TEST_T_ANNEAL_NS = 10.0
TEST_NUM_READS = 100


def main():
    schedule = config.load_advantage2_schedule()
    samplers, embeddings, base = get_sampler()
    extended_j_range = tuple(base.properties["extended_j_range"])
    print(f"\nget_sampler() OK -- {len(samplers)} topologies embedded.\n")

    test_points = [
        ("ring", 1.0, 1),
        ("native_e5_c1_r0", 1.0, None),
        ("native_e6_c26_r0", 1.0, 1),
    ]

    for topology, j_multiple, j_sign in test_points:
        print(f"--- {topology}  J={j_multiple:g}x  sign={j_sign} ---")
        row = run_point(samplers[topology], embeddings[topology], schedule, topology, j_multiple,
                         TEST_T_ANNEAL_NS, TEST_NUM_READS, extended_j_range, j_sign=j_sign)
        for k, v in row.items():
            print(f"  {k}: {v}")
        print()

    print("Smoke test complete -- all 3 code paths ran without error.")


if __name__ == "__main__":
    main()
