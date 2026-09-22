"""Score picky's read of the same strips against the reader's, on one metric."""
import sys
import wave

import numpy as np

from paper_sound import BASELINE, highpass, read_wav, to_rate


def any_wav(path):
    """read_wav, but 24-bit as well -- picky writes 24 and paper_sound 16."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 3:
            return read_wav(path)
        raw = np.frombuffer(w.readframes(w.getnframes()), np.uint8)
        b = raw.reshape(-1, 3).astype(np.int32)
        x = (b[:, 0] | b[:, 1] << 8 | (b[:, 2] ^ 0x80) << 16) - (1 << 23)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
        return x / float(1 << 23), w.getframerate()


def best_lag(a, b, reach):
    """Offset of a inside b, by FFT cross-correlation. b is the longer one."""
    n = 1 << (len(a) + len(b)).bit_length()
    c = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    return int(np.argmax(c[:reach]))


def per_second(got, sr, ref, reach=40000):
    """r for each reference second the read covers -> {second: r}."""
    a, b = highpass(got, BASELINE), highpass(ref, BASELINE)
    # the read may start before the reference (picky's lead-in and head clock)
    d = best_lag(b, a, reach)                      # where the reference starts in the read
    out = {}
    # Sliced out of the whole-file high-pass, not high-passed per slice: taking
    # the baseline off each second separately quietly removes the per-lane
    # offset too, which is the very thing a wide lane gets wrong, and lifts
    # sheetC from 0.79 to 0.97. That is align.py's question, not score.py's.
    # Flat seconds are kept as NaN rather than dropped. Dropped, they leave
    # `common` in main() a different set of seconds for each reader, and a
    # reader whose failures were silence gets a better median for having
    # failed -- which is the one thing this file exists not to do.
    for k in range(len(ref) // sr):
        x, y = a[d + k * sr:d + (k + 1) * sr], b[k * sr:(k + 1) * sr]
        if len(x) < sr:
            break
        out[k] = (float(np.corrcoef(x, y)[0, 1])
                  if x.std() > 0 and y.std() > 0 else float("nan"))
    return d, out


def db(r):
    return 10 * np.log10(r * r / (1 - r * r)) if abs(r) < 1 else float("inf")


def run(label, path, sr, ref_path):
    got, _ = any_wav(path)
    ref, rr = any_wav(ref_path)
    ref = to_rate(ref, rr, sr)
    d, per = per_second(got, sr, ref, )
    return label, d, per


def main(sr, ref_path, *pairs):
    sr = int(sr)
    runs = [run(lab, p, sr, ref_path) for lab, p in
            (pairs[i:i + 2] for i in range(0, len(pairs), 2))]
    common = set.intersection(*[set(per) for _, _, per in runs])
    print(f"against {ref_path} at {sr} Hz, {len(common)} seconds both cover")
    for lab, d, per in runs:
        v = np.array([per[k] for k in sorted(common)])
        flat = int((~np.isfinite(v)).sum())
        bad = ~np.isfinite(v) | (v < 0.5)
        print(f"  {lab:22s} lead {d:6d} samp  median r {np.nanmedian(v):.4f} "
              f"({db(np.nanmedian(v)):5.2f} dB)  mean {np.nanmean(v):.4f}  "
              f"below .5 {int(bad.sum()):3d}/{len(v)}"
              f"{f' ({flat} flat)' if flat else ''}  "
              f"worst {' '.join(f'{x:.2f}' for x in np.sort(v)[:3])}")


if __name__ == "__main__":
    main(*sys.argv[1:])
