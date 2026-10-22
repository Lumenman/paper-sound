#!/usr/bin/env python3
"""What averaging buys: several scans of one sheet, and the lanes of one row.

    python diversity.py scanA.png scanB.png scanC.png scanD.png song.wav

The radio trick, two antennas: add what the two share if their noise is
independent, subtract what they share if the signal is not. On paper the
antennas are the scans of one sheet (add) and the lanes of one row (subtract a
reference stroke's view of the scanner). Both only pay as far as the noise is
independent, so this prints the correlations as well as the dB.

Measured on sheetD (LAB 29): the error of two scans of one sheet correlates at
rho 0.61-0.77, so four scans buy +1.0 dB and no number of them more than
+1.4; lanes of one row share 6.7% of their error power against 1.7% by
chance, so a perfect reference stroke would buy +0.22 dB.

Reads are brought to the wav's rate rather than the wav to each read's, since
scans of one sheet come back at different rates (6576-6608 Hz on sheetD) and
have to land on one grid to be averaged. That costs a single scan about 1 dB
against bench.py; it costs the comparison nothing.
"""
import itertools
import sys

import numpy as np

from paper_sound import (BASELINE, DECLICK, align, declick, highpass,
                         load_ink, pilot_retime, read_curves_page, read_wav,
                         to_rate)


def db(r):
    return 10 * np.log10(r * r / (1 - r * r))


def pair(scan, truth, lag=8):
    """-> (read, ref, rate): aligned, at the wav's rate, unit variance."""
    song, sr, n = read_curves_page(load_ink(scan))
    song, n, js, turned, rows = pilot_retime(song, sr, n)
    sr = (sr // rows) * rows
    joins = np.arange(1, round(len(song) / (sr // rows))) * (sr // rows)
    song = declick(highpass(song, BASELINE), joins, DECLICK)
    ref, rate = read_wav(truth)
    song, ref = to_rate(song, sr, rate), highpass(ref, BASELINE)
    _, a, b = align(song, ref, lag)
    z = lambda x: (x - x.mean()) / x.std()
    return z(a), z(b), rate


def err(a, b):
    return a - np.dot(a, b) / np.dot(b, b) * b


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__.strip().split("\n\n")[1])
    *scans, truth = argv
    pairs = [pair(s, truth) for s in scans]
    n = min(len(a) for a, _, _ in pairs)
    A = [a[:n] for a, _, _ in pairs]
    b, sr = pairs[0][1][:n], pairs[0][2]
    for s, a in zip(scans, A):
        print(f"{s:28s} {db(np.corrcoef(a, b)[0, 1]):6.2f} dB")

    E = [err(a, b) for a in A]
    print("\nerror correlation between scans")
    for i, j in itertools.combinations(range(len(A)), 2):
        print(f"  {i + 1}-{j + 1}  rho={np.corrcoef(E[i], E[j])[0, 1]:.3f}")
    for k in range(2, len(A) + 1):
        vals = [db(np.corrcoef(np.mean([A[i] for i in c], 0), b)[0, 1])
                for c in itertools.combinations(range(len(A)), k)]
        print(f"  mean of {k}: median {np.median(vals):.2f} dB  "
              f"range {min(vals):.2f}..{max(vals):.2f}")

    # A lane is one second of audio, so sample i and i + sr are one row.
    m = n // sr
    print(f"\nerror correlation between lanes of one row ({m} lanes)")
    for name, e in [("first scan", E[0]), (f"mean of {len(E)}", np.mean(E, 0))]:
        lanes = e[:m * sr].reshape(m, sr)
        lanes = (lanes - lanes.mean(1, keepdims=True)) / lanes.std(1, keepdims=True)
        for dk in (1, 2, 5, 10):
            rho = np.mean([lanes[k] @ lanes[k + dk] / sr for k in range(m - dk)])
            print(f"  {name:10s} k,k+{dk:<2d}  rho={rho:+.4f}")
        rho = np.mean([lanes[k] @ np.roll(lanes[k + 1], sr // 4) / sr
                       for k in range(m - 1)])
        print(f"  {name:10s} rows shifted a quarter lane  rho={rho:+.4f}")
        frac = lanes.mean(0).var()
        print(f"  {name:10s} common to the row: {100 * frac:.2f}% of the error "
              f"(chance {100 / m:.2f}%); a reference stroke that took it all "
              f"out: {-10 * np.log10(1 - frac + 1 / m):+.2f} dB")


if __name__ == "__main__":
    main(sys.argv[1:])
