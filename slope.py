"""Read error against the slope the ink had to draw, in px per row.

The other benches say how much a read lost. This one says where: a sheet that
loses to its own steepness loses in the top bins only, a sheet that loses to
paper loses in all of them evenly.

    python slope.py rB.wav:src59.wav:38.7 rC.wav:src59.wav:77.4

The pitch is needed because slope is a property of the printed sheet, not of
the audio: the same wav at 77.4 px moves the stroke twice as many px per row.
"""
import sys

import numpy as np

from paper_sound import BASELINE, MARGIN, STROKE, highpass, read_wav, to_rate


def swing(pitch):
    """px of lane the stroke can use, printer's amp = (pitch - stroke)/2 - margin."""
    return pitch - STROKE - 2 * MARGIN


def report(got, truth, pitch, lag=64):
    x, sr = read_wav(got)
    y, rate = read_wav(truth)
    ref = to_rate(y, rate, sr)
    n = min(len(x), len(ref)) - 2 * lag
    a, b = highpass(x, BASELINE), highpass(ref, BASELINE)
    # Both ways: a paper read sits a sample or two late, and that is taken out
    # by advancing the read. One-sided it reads as noise -- see bench.score.
    d = int(np.argmax([np.corrcoef(a[lag + k:lag + k + n], b[lag:lag + n])[0, 1]
                       for k in range(-lag, lag + 1)])) - lag
    a, ref = a[lag + d:lag + d + n], b[lag:lag + n]
    a = a * (ref @ a) / (a @ a)               # r is scale-free; the error is not
    err = a - ref
    px = np.abs(np.gradient(ref)) / (2 * np.abs(ref).max()) * swing(pitch)
    print(f"{got} vs {truth}  (pitch {pitch} px, excursion {swing(pitch):.0f} px)")
    q = np.percentile(px, [0, 50, 75, 90, 97, 99.5, 100])
    for lo, hi in zip(q[:-1], q[1:]):
        m = (px >= lo) & (px < hi)
        if m.sum() < 100:
            continue
        e = 10 * np.log10((err[m] @ err[m]) / (ref[m] @ ref[m]))
        print(f"   slope {lo:5.1f}-{hi:5.1f} px/row  {m.sum():7d} samples   "
              f"err/sig {e:+6.2f} dB")


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit(__doc__.strip().split("\n\n")[1])
    for arg in sys.argv[1:]:
        g, r, p = arg.rsplit(":", 2)
        report(g, r, float(p))
