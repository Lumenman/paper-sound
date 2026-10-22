import sys
import numpy as np
from paper_sound import BASELINE, align, highpass, read_wav, to_rate

def score(got_path, ref_path, lag=64):
    got, sr = read_wav(got_path)
    ref, rate = read_wav(ref_path)
    ref = to_rate(ref, rate, sr)
    # Both ways and to a fraction of a sample, see paper_sound.align().
    r, a, b = align(highpass(got, BASELINE), highpass(ref, BASELINE), lag)
    n = len(b)
    per = np.array([np.corrcoef(a[i:i+sr], b[i:i+sr])[0, 1]
                    for i in range(0, n - sr + 1, sr)])
    db = 10 * np.log10(r * r / (1 - r * r))
    # NaN is a lane that came back as silence, not a lane that scored zero:
    # `NaN < 0.5` is False, so counted straight it is the worst second on the
    # page hiding inside the pass count. See bench.py.
    flat = int((~np.isfinite(per)).sum())
    bad = ~np.isfinite(per) | (per < 0.5)
    print(f"{got_path:12s} vs {ref_path:10s} r={r:.4f}  {db:6.2f} dB  "
          f"secs {len(per)}  median {np.nanmedian(per):.4f}  "
          f"below .5 {bad.sum()}/{len(per)}{f' ({flat} flat)' if flat else ''}  "
          f"worst {' '.join(f'{v:.2f}' for v in np.sort(per)[:3])}")

if __name__ == "__main__":
    for pair in sys.argv[1:]:
        score(*pair.rsplit(":", 1))
