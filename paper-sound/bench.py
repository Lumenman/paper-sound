#!/usr/bin/env python3
"""Score a read of a scan against the audio that went onto the paper.

The selftest in paper_sound.py checks a sheet that never left the computer.
This checks one that did, which is the only measurement that means anything --
a printer and a flatbed do things no simulation of them predicted correctly.

    python bench.py sheet.png song.wav
    python bench.py sheet_01.png sheet_02.png song.wav
    python bench.py --same scanA.png scanB.png scanC.png song.wav
    python bench.py --rows=3 sheet.png song.wav

The last argument is the wav that was printed; everything before it is the
scans, in playing order. It prints one line:

    sheet.png   r=0.9383  lanes  57 at 6574 Hz  2 clocks  median 0.9354
                below 0.5: 1/56  worst 0.28 0.79 0.82
                below 0.5 at second 31

`--rows=` is `read --rows`: a sheet printed at 3 or 4 rows a sample says so
only if asked, see CLOCK_ROWS in paper_sound.py.

`--same` says the scans are one sheet scanned more than once, rather than the
pages of one recording, and adds a median line under them. It is there because
one scan does not measure a sheet: four scans of sheetD, same settings, spread
over 1.68 dB, and the better of two taken minutes apart was the one whose
carriage rippled worse. Under the old whole-sample lag that spread looked like
1.5-2 dB and more, much of it the count rather than the scan (LAB 29.10); on
the sheets since it is 0.5-0.7. Anything smaller than the spread is the
scanner talking, not the sheet -- so a difference worth reporting is a
difference between medians of three, and the spread printed beside the median
says whether the number is even worth having.

Scan the three with the sheet lifted and laid down again between them. Left on
the glass it is the same placement three times, and placement is half of what
is being averaged out.

The seconds are listed because WHERE they are is the diagnosis. Scattered
singly across the page they are a per-lane fault -- look at the strips
cut_lanes.py writes for those seconds. In one long run they are the paper: a
fold, a band of ripple, the edge of the glass. Measured on the half-pitch sheet
here, 27 bad seconds came back as 1-2, 13, 32, 44, 73 ... 237-239, which is
scattered, and the sheet is fine everywhere else.

One global lag is taken out before anything is compared, searched BOTH ways.
Print and scan fade the ink's first row into the paper, so a read sits a sample
or two late against the file uniformly; that is not a defect of the read, and
on broadband audio it is worth 0.15 of correlation. Late means the READ is
advanced, not the reference -- searching one way only is what this file used to
do, against its own paragraph here, and it cost sheetB 4.8 dB and sheetC 10.4,
which then read as a difference between the two formats and was not one. The
lag is then refined to 1/32 of a sample: half a sample left in cost up to 2.4
dB at rows 1 and most of the spread between scans (LAB 29.9), and a constant
delay that small is not heard. Per-lane offsets are NOT taken out: those are
real, and taking them out is what the clock lanes do.

The lane distribution matters more than the total. A reader that loses two
seconds out of a hundred barely moves the total and is very audible, so the
count below 0.5 and the worst few are printed as well.

What to expect. A full sheet straight out of `print`, never on paper, reads
back at r > 0.999 -- less than that means the reader is broken rather than the
paper. A partial sheet scores lower for an uninteresting reason: its last lane
is padded with silence and correlates with nothing.
Printed and scanned, the sheets here come back at 0.92 to 0.97, and what
separates them is the scanner rather than the format: how crooked the sheet lay
and how badly the carriage rippled that day. Below about 0.85 something is
worth finding, and the warnings `read` prints are where to start looking --
`--aperture` first if the sheet's pitch is not the default one, since the
window is tuned for 38.7 px and a wider lane wants it wider (6.0 dB on the
77.4 px sheet here).
"""
import os
import sys

import numpy as np

from paper_sound import (BASELINE, DECLICK, align, declick, highpass,
                         load_ink, pilot_retime, read_curves_page, read_wav,
                         retouched, to_rate)


def runs(ns, cap=16):
    """[1, 2, 13, 209, 210] -> "1-2, 13, 209-210"."""
    out, start = [], ns[0]
    for a, b in zip(ns, list(ns[1:]) + [None]):
        if b != a + 1:
            out.append(f"{start}-{a}" if a > start else f"{start}")
            start = b
    return ", ".join(out[:cap]) + (" ..." if len(out) > cap else "")


def score(song, sr, truth, hp=BASELINE, lag=8):
    """(overall r, per-lane r) against the printed audio, at the read's rate."""
    ref, rate = read_wav(truth)
    ref = to_rate(ref, rate, sr)
    if min(len(song), len(ref)) - 2 * lag < sr:
        raise SystemExit("the read and the reference barely overlap")
    # Both ways and to a fraction of a sample, see align(). The digital sheets
    # peak at 0 either way, which is what says the search is finding the fade
    # and not fitting noise.
    r, a, b = align(highpass(song, hp), highpass(ref, hp), lag)
    n = len(b)
    lanes = np.array([np.corrcoef(a[i:i + sr], b[i:i + sr])[0, 1]
                      for i in range(0, n - sr + 1, sr)])
    return r, lanes


def main(argv):
    same = "--same" in argv
    given = [int(a.split("=", 1)[1]) for a in argv if a.startswith("--rows=")]
    argv = [a for a in argv if a != "--same" and not a.startswith("--rows=")]
    if len(argv) < 2:
        raise SystemExit(__doc__.strip().split("\n\n")[2])
    *scans, truth = argv
    for path in (*scans, truth):
        if not os.path.exists(path):
            raise SystemExit(f"{path}: no such file")

    rs = []
    for path in scans:
        ink = load_ink(path)
        if retouched(ink):
            print(f"{path}: NOTE: levels adjusted after scanning")
        song, sr, n = read_curves_page(ink)
        song, n, js, turned, rows = pilot_retime(song, sr, n,
                                                 rows=given[0] if given else None)
        sr = (sr // rows) * rows     # a thinned lane's own rate; see thin()
        if turned:
            print(f"{path}: NOTE: scanned upside down; the read was turned")
        # Scored through the same post-processing a read applies, so that this
        # number and the wav on disk are the same read.
        joins = np.arange(1, round(len(song) / (sr // rows))) * (sr // rows)
        song = declick(highpass(song, BASELINE), joins, DECLICK)
        r, lanes = score(song, sr, truth)
        # A lost lane is the failure this line is for, and it was the one
        # failure it could not see: read_curves_page pads it with silence,
        # correlation against silence is NaN, and `NaN < 0.5` is False -- so
        # the second that went missing entirely was counted as passing, and
        # one of them turned the median into NaN as well.
        flat = int((~np.isfinite(lanes)).sum())
        bad = ~np.isfinite(lanes) | (lanes < 0.5)
        print(f"{path:20s} r={r:.4f}  lanes {n:3d} at {sr} Hz  "
              f"{'' if rows == 1 else f'{rows} rows a sample  '}"
              f"{len(js)} clock{'s' if len(js) != 1 else ''}  "
              f"median {np.nanmedian(lanes):.4f}  "
              f"below 0.5: {bad.sum():3d}/{len(lanes)}"
              f"{f' ({flat} flat)' if flat else ''}  "
              f"worst {' '.join(f'{v:.2f}' for v in np.sort(lanes)[:3])}")
        rs.append(r)
        bad = np.flatnonzero(bad) + 1
        if len(bad):
            print(f"{'':20s} below 0.5 at second{'s' if len(bad) != 1 else ''} "
                  f"{runs(list(bad))}")

    if same and len(rs) > 1:
        db = [10 * np.log10(v * v / (1 - v * v)) for v in rs]
        print(f"{'median of ' + str(len(rs)) + ' scans':20s} "
              f"r={np.median(rs):.4f}  {np.median(db):5.2f} dB   "
              f"spread {min(rs):.4f} to {max(rs):.4f}, "
              f"{max(db) - min(db):.2f} dB")


if __name__ == "__main__":
    main(sys.argv[1:])
