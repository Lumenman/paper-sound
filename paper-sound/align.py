"""Per-second r with a per-second lag search: shows where a read slipped."""
import sys
import numpy as np
from paper_sound import BASELINE, highpass, read_wav, to_rate


def norm_xcorr(x, y):
    """r of x against every window of y, FFT, unit-normalised per window."""
    n, m = len(x), len(y)
    x = x - x.mean()
    L = 1 << (m + n).bit_length()
    c = np.fft.irfft(np.fft.rfft(y, L) * np.conj(np.fft.rfft(x, L)), L)[:m - n + 1]
    cs, cs2 = np.cumsum(np.r_[0, y]), np.cumsum(np.r_[0, y * y])
    s = cs[n:] - cs[:-n + 1 or None][:len(c)]
    s2 = cs2[n:] - cs2[:len(c)]
    var = np.maximum(s2 - s * s / n, 1e-30)
    return (c - s * x.mean()) / (np.sqrt(var) * np.sqrt(x @ x) + 1e-30)


def run(got_path, ref_path, span=3.0):
    got, sr = read_wav(got_path)
    ref, rate = read_wav(ref_path)
    ref = to_rate(ref, rate, sr)
    a, b = highpass(got, BASELINE), highpass(ref, BASELINE)
    w = int(span * sr)
    out = []
    for i in range(0, len(a) - sr + 1, sr):
        x = a[i:i + sr]
        lo, hi = max(0, i - w), min(len(b), i + sr + w)
        if hi - lo < sr + 2:
            break
        if x.std() < 1e-12:
            # A silent second has no lag to find, but it is not the end of the
            # file: this used to `break`, so one pause in the music stopped the
            # walk and every second after it went unreported as if clean.
            out.append((i // sr, float("nan"), float("nan")))
            continue
        c = norm_xcorr(x, b[lo:hi])
        j = int(np.argmax(c))
        out.append((i // sr, float(c[j]), (lo + j - i) / sr))
    return out


if __name__ == "__main__":
    for pair in sys.argv[1:]:
        g, r = pair.rsplit(":", 1)
        res = run(g, r)
        rr = np.array([v for _, v, _ in res])
        lg = np.array([d for _, _, d in res])
        print(f"{g} vs {r}: {len(res)} s, median r {np.nanmedian(rr):.4f}")
        print("  r    :", " ".join(f"{v:.2f}" for v in rr))
        print("  lag s:", " ".join(f"{v:+.2f}" for v in lg))
