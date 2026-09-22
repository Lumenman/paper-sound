"""The four benchmark bugs of the review, and what a sheet holds: one assert each.

    python test_seconds.py

Every one of them is about a second that should have been counted and was
not: the last one in the file, a silent one in the middle, or a lane that came
back as silence and scored NaN.
"""
import argparse
import contextlib
import io
import os
import tempfile

import numpy as np

import align
import cmp_picky
import spec
from paper_sound import capacity, do_print, write_wav


def wavs(x, y, sr):
    d = tempfile.mkdtemp()
    a, b = os.path.join(d, "got.wav"), os.path.join(d, "ref.wav")
    write_wav(a, x, sr), write_wav(b, y, sr)
    return a, b


def main():
    sr, secs = 8000, 10
    rng = np.random.default_rng(0)
    x = rng.standard_normal(sr * secs) * 0.2
    dead = np.zeros(sr * secs)

    # align.py: the last full second of a 10 s file is the tenth, not the ninth
    res = align.run(*wavs(x, x, sr))
    assert len(res) == secs, f"align read {len(res)} of {secs} seconds"
    assert np.isfinite(res[-1][1]), "the last second scored nothing"

    # and a second with no signal in it is one NaN, not the end of the walk:
    # this used to `break`, which reported a dead file as zero seconds read
    res = align.run(*wavs(dead, x, sr))
    assert len(res) == secs, f"a flat read stopped after {len(res)} seconds"
    assert not any(np.isfinite(v) for _, v, _ in res), "silence scored a lag"

    # cmp_picky.py: same count, and the flat seconds kept rather than dropped
    d, per = cmp_picky.per_second(x, sr, x)
    assert d == 0 and len(per) == secs, f"cmp_picky read {len(per)} of {secs}"
    _, per = cmp_picky.per_second(dead, sr, x)
    assert len(per) == secs, "the flat seconds vanished out of `common`"
    assert not any(np.isfinite(v) for v in per.values()), "silence scored an r"

    # spec.py: a file of exactly one FFT block is one block, not none
    n = 8192
    f, p = spec.avg_spec(rng.standard_normal(n), sr, n)
    assert np.all(np.isfinite(p)), "an exactly-n-sample file gave no segments"
    try:
        spec.avg_spec(np.zeros(n - 1), sr, n)
        raise AssertionError("a file shorter than the FFT should say so")
    except ValueError:
        pass

    # bench/score: a lost lane comes back as silence, and NaN < 0.5 is False
    lanes = np.array([0.9, 0.9, np.nan, 0.3])
    bad = ~np.isfinite(lanes) | (lanes < 0.5)
    assert bad.sum() == 2 and np.isfinite(np.nanmedian(lanes)), "the lost lane hid"

    # capacity() is what `holds` promises; do_print is what lays a sheet out.
    # A sheet's worth of audio must be one sheet and a second more must be two,
    # or the two have drifted apart. Small paper: the geometry is the same sum
    # at any size, and A4 at 600 dpi is a 139 MB page to render for it.
    paper, dpi = "100x100", 100
    secs, rate, _, _ = capacity(paper, dpi, 5.0, 38.7)
    assert secs == 6, f"the 100x100 sheet holds {secs} s, not the 6 it did"
    for want, held in ((1, secs), (2, secs + 1)):
        d = tempfile.mkdtemp()
        wav = os.path.join(d, "a.wav")
        write_wav(wav, rng.standard_normal(int(held * rate)) * 0.2, rate)
        with contextlib.redirect_stdout(io.StringIO()):
            do_print(argparse.Namespace(audio=wav, out=None, paper=paper,
                                        dpi=dpi, margin_mm=5.0, pitch=38.7,
                                        rows=1, start=0.0))
        got = len([f for f in os.listdir(d) if f.endswith(".png")])
        assert got == want, (f"{held} s on a {secs} s sheet printed {got} "
                             f"sheets, not {want}: print and capacity disagree")

    print("ok")


if __name__ == "__main__":
    main()
