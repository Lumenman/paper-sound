import sys
import numpy as np
from paper_sound import BASELINE, highpass, read_wav, to_rate

def score(got_path, ref_path, lag=64):
    got, sr = read_wav(got_path)
    ref, rate = read_wav(ref_path)
    ref = to_rate(ref, rate, sr)
    n = min(len(got), len(ref)) - 2 * lag
    a, b = highpass(got, BASELINE), highpass(ref, BASELINE)
    # Searched both ways: a paper read sits a sample or two LATE (the ink's
    # first row fades into the paper), and that is taken out by advancing the
    # read, not the reference. One-sided, it cost sheetC 10.4 dB of nothing.
    rs = [np.corrcoef(a[lag + d:lag + d + n], b[lag:lag + n])[0, 1]
          for d in range(-lag, lag + 1)]
    r = max(rs)
    a = a[lag + int(np.argmax(rs)) - lag:][:n]
    b = b[lag:lag + n]
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

for pair in sys.argv[1:]:
    score(*pair.rsplit(":", 1))
