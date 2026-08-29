"""Where a read has energy the printed audio did not."""
import sys
import numpy as np
from paper_sound import BASELINE, highpass, read_wav, to_rate

def avg_spec(x, sr, n=8192):
    w = np.hanning(n)
    segs = [np.abs(np.fft.rfft(x[i:i+n] * w)) ** 2
            for i in range(0, len(x) - n, n // 2)]
    return np.fft.rfftfreq(n, 1 / sr), np.mean(segs, 0)

def report(got, ref, label, top=8):
    x, sr = read_wav(got)
    y, rate = read_wav(ref)
    y = to_rate(y, rate, sr)
    n = min(len(x), len(y))
    f, px = avg_spec(highpass(x[:n], BASELINE), sr)
    _, py = avg_spec(highpass(y[:n], BASELINE), sr)
    ex = 10 * np.log10((px + 1e-30) / (py + 1e-30))
    ex -= np.median(ex)                       # broadband tilt is not the point
    band = (f > 40) & (f < sr / 2 - 40)
    idx = np.argsort(ex[band])[::-1]
    fb, eb = f[band], ex[band]
    peaks, seen = [], []
    for i in idx:
        if any(abs(fb[i] - s) < 30 for s in seen):
            continue
        seen.append(fb[i]); peaks.append((fb[i], eb[i]))
        if len(peaks) == top:
            break
    print(f"{label} ({sr} Hz)")
    for fr, e in sorted(peaks):
        print(f"    {fr:8.1f} Hz  {e:+6.1f} dB over the source   "
              f"(a feature every {sr/fr:6.2f} rows)")

for a in sys.argv[1:]:
    g, r, lab = a.rsplit(":", 2)
    report(g, r, lab)

def bands(got, ref, label, step=200):
    x, sr = read_wav(got); y, rate = read_wav(ref); y = to_rate(y, rate, sr)
    n = min(len(x), len(y))
    f, px = avg_spec(highpass(x[:n], BASELINE), sr)
    _, py = avg_spec(highpass(y[:n], BASELINE), sr)
    ex = 10*np.log10((px+1e-30)/(py+1e-30)); ex -= np.median(ex)
    print(f"{label:22s}", end="")
    for lo in range(0, int(sr/2), step):
        m = (f >= lo) & (f < lo+step)
        print(f"{np.median(ex[m]):+6.1f}", end="")
    print()
