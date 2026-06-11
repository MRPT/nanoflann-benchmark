#!/usr/bin/env python3
"""Compare the manifold-on vs manifold-off CSVs: assert identical checksums
and print build/query time ratios (on / off)."""
import csv
import sys


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[int(r["N"])] = r
    return rows


def main():
    on = load(sys.argv[1])
    off = load(sys.argv[2])
    assert on.keys() == off.keys()
    print(f"{'N':>10} {'build on/off':>13} {'query on/off':>13}  identical")
    ok = True
    for n in sorted(on):
        rb = float(on[n]["build_s"]) / float(off[n]["build_s"])
        rq = float(on[n]["query_us"]) / float(off[n]["query_us"])
        same = on[n]["checksum"] == off[n]["checksum"]
        ok = ok and same
        print(f"{n:>10} {rb:>13.3f} {rq:>13.3f}  {'YES' if same else 'NO!'}")
    if not ok:
        print("ERROR: result checksums differ between builds", file=sys.stderr)
        sys.exit(1)
    print("All checksums identical: results are bitwise-equal.")


if __name__ == "__main__":
    main()
