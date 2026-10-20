#!/usr/bin/env python3
"""Does the paper's error depend on the signal, and can a reader take it out?

    python sigdep.py fitA.png fitB.png -- testC.png testD.png song.wav

Fits the error of the scans before `--` as a function of the local stroke
shape (value, slope, curvature and nonlinear functions of them, +-4 samples)
and takes the prediction out of the scans after it. Twice: with the shape
taken from the wav, which says how much of the error is predictable at all,
and from the read, which is all a reader would have.

Measured on sheetD, sheetB and sheetC (LAB 29.3): from the wav +2-3 dB, and it
carries from one sheet to another; nearly all of it is linear, the chain's own
treble loss. From the read +0.1 to +0.8, and below zero fitted on sheetB --
the treble lies under the paper's noise, which is what the inverse filter
found too. So a calibration file for the reader is not worth building.

`--kind=lin` or `--kind=non` restricts the features, to tell the two apart.

Last, a Wiener filter: the spectra of signal and paper noise of the fit scans,
and the read turned down wherever the paper is the louder. It is the noise
print a known tone or a blank stretch of paper would give. Measured (LAB 29.5):
+0.04 to +0.16 dB, because above 1600 Hz the read and the paper are equally
loud and turning one down turns down the other.
"""
import sys

import numpy as np
from scipy.signal import welch

from diversity import db, err, pair


def wiener(pairs, sr, nper=512):
    S = sum(welch(a - err(a, b), sr, nperseg=nper)[1] for a, b in pairs)
    f, N = welch(np.concatenate([err(a, b) for a, b in pairs]), sr, nperseg=nper)
    N = N * len(pairs)
    return f, S / (S + N)


def filtered(a, f, H, sr):
    return np.fft.irfft(np.fft.rfft(a) * np.interp(np.fft.rfftfreq(len(a), 1 / sr), f, H), len(a))


def feats(x, kind="all", taps=4):
    d = np.gradient(x)
    c = np.gradient(d)
    lin = [x, d, c]
    non = [d * np.abs(d), d ** 2, d ** 3, np.abs(d), c * np.abs(c), x * np.abs(x)]
    base = {"lin": lin, "non": non, "all": lin + non}[kind]
    F = np.stack([np.roll(f, k) for f in base for k in range(-taps, taps + 1)], 1)
    return F - F.mean(0)


def fit(pairs, src, kind):
    X = np.concatenate([feats(p[src], kind) for p in pairs])
    e = np.concatenate([err(a, b) for a, b in pairs])
    lam = 1e-3 * np.trace(X.T @ X) / X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ e)


def main(argv):
    kind = next((a.split("=", 1)[1] for a in argv if a.startswith("--kind=")), "all")
    argv = [a for a in argv if not a.startswith("--kind=")]
    if "--" not in argv or len(argv) < 4:
        raise SystemExit(__doc__.strip().split("\n\n")[1])
    cut = argv.index("--")
    *rest, truth = argv[cut + 1:]
    fits, tests = argv[:cut], rest
    sr = pair(fits[0], truth)[2]
    P = {s: pair(s, truth)[:2] for s in fits + tests}
    for src, name in ((1, "the wav"), (0, "the read")):
        w = fit([P[s] for s in fits], src, kind)
        print(f"shape from {name}, {kind} features")
        for s in tests:
            a, b = P[s]
            before = db(np.corrcoef(a, b)[0, 1])
            after = db(np.corrcoef(a - feats(P[s][src], kind) @ w, b)[0, 1])
            print(f"  {s:28s} {before:6.2f} -> {after:6.2f} dB  ({after - before:+.2f})")
    f, H = wiener([P[s] for s in fits], sr)
    print("Wiener filter from the fit scans' spectra")
    for s in tests:
        a, b = P[s]
        before, after = db(np.corrcoef(a, b)[0, 1]), db(np.corrcoef(filtered(a, f, H, sr), b)[0, 1])
        print(f"  {s:28s} {before:6.2f} -> {after:6.2f} dB  ({after - before:+.2f})")


if __name__ == "__main__":
    main(sys.argv[1:])
