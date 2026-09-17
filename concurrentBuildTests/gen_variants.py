#!/usr/bin/env python3
"""
Generate derived copies of a nanoflann header for the concurrent-build study.

Three kinds of variant are produced under <outdir>/:

  cutoff<N>/nanoflann.hpp   kDivideConcurrentTaskCutoff overridden to N, to check
                            whether the hard-coded task-size threshold is a good
                            default.
  count/nanoflann.hpp       increments a global counter every time a task is
                            spawned, so a build can report spawns/build.
  peak/nanoflann.hpp        tracks the peak and minimum value of tasks_in_flight,
                            to check that the advertised "at most n_thread_build_
                            threads" bound actually holds.

The instrumented variants only apply to the per-task-pool builder (the rewritten
divideTreeConcurrent); against an older header they are skipped with a notice.

Usage:
  gen_variants.py <path/to/nanoflann.hpp> <outdir> [cutoffs...]
"""

import os
import sys

DEFAULT_CUTOFFS = [512, 2048, 8192, 32768]

CUTOFF_ANCHOR = "kDivideConcurrentTaskCutoff = 512;"
SPAWN_ANCHOR = "                    spawned = true;"
DECR_ANCHOR = "            tasks_in_flight.fetch_sub(1, std::memory_order_acq_rel);"
NS_ANCHOR = "namespace nanoflann\n{"

COUNT_DECL = "namespace nanoflann\n{\nextern std::atomic<long> g_nf_spawns;\n"
COUNT_BODY = """                    spawned = true;
                    g_nf_spawns.fetch_add(1, std::memory_order_relaxed);"""

PEAK_DECL = (
    "namespace nanoflann\n{\n"
    "extern std::atomic<int>  g_nf_peak;\n"
    "extern std::atomic<int>  g_nf_min;\n"
    "extern std::atomic<long> g_nf_threads;\n"
)
PEAK_SPAWN = """                    spawned = true;
                    {
                        const int now = expected + 1;
                        int prev = g_nf_peak.load();
                        while (now > prev && !g_nf_peak.compare_exchange_weak(prev, now)) {}
                        g_nf_threads.fetch_add(1);
                    }"""
PEAK_DECR = """            {
                const int after = tasks_in_flight.fetch_sub(1, std::memory_order_acq_rel) - 1;
                int prev = g_nf_min.load();
                while (after < prev && !g_nf_min.compare_exchange_weak(prev, after)) {}
            }"""


def write(outdir, name, text):
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "nanoflann.hpp"), "w") as f:
        f.write(text)
    print(f"  {name}/nanoflann.hpp")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    header, outdir = sys.argv[1], sys.argv[2]
    cutoffs = [int(c) for c in sys.argv[3:]] or DEFAULT_CUTOFFS

    src = open(header).read()
    print(f"Generating variants of {header} into {outdir}:")

    if CUTOFF_ANCHOR not in src:
        print("  (no kDivideConcurrentTaskCutoff: not a per-task-pool builder, skipping)")
        return 0

    for c in cutoffs:
        write(outdir, f"cutoff{c}",
              src.replace(CUTOFF_ANCHOR, f"kDivideConcurrentTaskCutoff = {c};"))

    assert src.count(SPAWN_ANCHOR) == 1, "spawn site not found exactly once"
    assert src.count(DECR_ANCHOR) == 1, "tasks_in_flight decrement not found exactly once"

    write(outdir, "count",
          src.replace(NS_ANCHOR, COUNT_DECL, 1).replace(SPAWN_ANCHOR, COUNT_BODY))

    write(outdir, "peak",
          src.replace(NS_ANCHOR, PEAK_DECL, 1)
             .replace(SPAWN_ANCHOR, PEAK_SPAWN)
             .replace(DECR_ANCHOR, PEAK_DECR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
