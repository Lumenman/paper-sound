#!/usr/bin/env python3
"""Read a lane by its stroke's symmetry axis instead of the ink's centroid.

    python axis.py sheetD_scan3.png src59.wav
    python axis.py --curve=40 sheetD_scan3.png src59.wav

The centroid is blind to how much ink landed only while the scanner's response
is LINEAR: it is a first moment weighted by intensity, so anything that bends
the greyscale reweights it and moves the answer. That is the hole the README
names and `retouched()` can only point at.

The symmetry axis does not have that hole. Walk out from the stroke's peak to
where the profile crosses a given level, left and right, and take the midpoint;
a monotone tone curve maps level sets to level sets, so every crossing lands on
the same pixels it did before. Average the midpoints over several levels and
the estimate is invariant to any monotone curve, exactly, up to the linear
interpolation used for the sub-pixel part.

The levels are taken as ORDER STATISTICS of the row's own profile -- the tenth
largest value in the window, and so on -- not as fixed greyscale values. That
is the whole trick and it is worth being explicit about: a monotone curve
permutes greyscale values but cannot reorder them, so a rank survives it and a
value does not. Streule's original walks fixed intensities from min+10 to
max-10, which is invariant level by level but reweights the average when the
curve bends; ranks close that gap.

After SymmetryAxisPE in Patrick Streule, "Digital Image Based Restoration of
Optical Movie Soundtracks", diploma thesis, ETHZ 1999 (comopt, GPL-3). His
estimator averages over subblocks of 300 film lines, where a track's position
moves slowly. Here one row IS one sample and nothing may be averaged down the
page, so the support is a single row across a 4 px stroke -- which is why this
is an experiment beside the reader and not the reader's default.
"""
import sys

import numpy as np

from bench import score
from paper_sound import (BASELINE, DECLICK, STROKE, declick, highpass,
                         lane_centroid, load_ink, pilot_retime,
                         read_curves_page, retouched)

WINDOW = 4 * STROKE   # px of lane the axis looks at: wider than the centroid's
                      # aperture on purpose. A first moment pays for every px
                      # of bare paper in its window; a level crossing does not,
                      # because the levels that matter all sit on the stroke.
TOPS = (1, 2, 3, 4, 5)  # which ranks, counted down from the row's brightest px,
                      # are used as levels. The ceiling is how many pixels the
                      # stroke actually covers: a 4 px stroke lights 5 or 6, the
                      # top one crosses nowhere, and rank 6 is already bare
                      # paper -- whose level crosses at the window's own edges
                      # and measures the window instead of the stroke.


def crossings(prof, level):
    """Sub-pixel columns where each row's profile crosses its own level.

    prof: (rows, w) ink profile, level: (rows,). Returns (left, right), the
    outermost crossings of the run of pixels at or above the level.
    """
    rows, w = prof.shape
    at = prof >= level[:, None]
    i = np.arange(rows)
    first = np.argmax(at, axis=1)
    last = w - 1 - np.argmax(at[:, ::-1], axis=1)

    def edge(inside, outside):
        # Interpolate into the pixel outside the run. At the window's own edge
        # there is no outside pixel and the crossing is clamped to it, which
        # leans that row towards the edge -- the caller throws those rows away
        # rather than correcting them.
        hi = prof[i, inside]
        lo = np.where((outside >= 0) & (outside < w),
                      prof[i, np.clip(outside, 0, w - 1)], hi)
        gap = np.maximum(hi - lo, 1e-9)
        step = np.clip((hi - level) / gap, 0, 1)
        return inside + step * np.sign(outside - inside)

    return edge(first, first - 1), edge(last, last + 1)


def lane_axis(ink, x0, x1, span, drift, narrow=None, sticky=None,
              width=None, tops=None):
    """Symmetry-axis position of one lane's stroke, per row. lane_centroid's twin.

    The centroid is asked for the AIM only -- where the window goes -- and the
    sample is measured from scratch inside it, in the page's own columns. So
    the aim's own bias moves the window by a fraction of a pixel and does not
    ride into the answer.
    """
    aimed = lane_centroid(ink, x0, x1, span, drift, narrow, sticky,
                          absolute=True)
    rel = lane_centroid(ink, x0, x1, span, drift, narrow, sticky)
    if aimed is None or rel is None:
        return None
    rows, centre = aimed
    # The window centre with the audio taken back out: x0 + span/2 + the skew
    # ramp, to a constant. Measuring against THAT is how the skew leaves the
    # samples -- the window rides it, so it subtracts exactly, and the sheet's
    # lean stops being audio. Against the trace's own median instead, the whole
    # ramp stays in, and on a crooked sheet it is as large as the recording:
    # sheetD_scan3 read 0.6887 that way against 0.9785 for the centroid, which
    # looked like a verdict on the estimator and was a verdict on this line.
    geom = centre - rel
    lane = ink[rows[0]:rows[1]].astype(np.float64)

    w = int(round(width if width else WINDOW * span / 38.7))
    w += w % 2                              # even, so the window has a centre
    start = np.clip(np.round(centre - w / 2).astype(int), 0, lane.shape[1] - w)
    prof = lane[np.arange(len(lane))[:, None], start[:, None] + np.arange(w)]

    order = np.sort(prof, axis=1)
    # Ranks, not greyscale values: what survives a tone curve. Counted from the
    # TOP, because the stroke is a tenth of its lane and a rank below the top
    # few is bare paper. Rank 0 is skipped too -- the maximum crosses nowhere
    # and its midpoint is just argmax, quantised to whole pixels.
    #
    # A rank is dropped ROW BY ROW rather than by picking a safe count once:
    # how many pixels the stroke lights is a property of the row (a moving
    # stroke smears over more of them, a crushed scan over fewer), so the same
    # rank is ink on one row and paper on the next. Two ways to be useless, and
    # both mean the level is not on a flank: it sits at the window's own floor,
    # or its run reaches an edge of the window and the crossing there is
    # clamped rather than measured.
    mids = np.full((len(np.atleast_1d(tops or TOPS)), len(prof)), np.nan)
    for j, t in enumerate(np.atleast_1d(tops or TOPS)):
        k = int(np.clip(w - 1 - t, 1, w - 2))
        level = order[:, k]
        lo, hi = crossings(prof, level)
        ok = ((level > order[:, 0])
              & (prof[:, 0] < level) & (prof[:, -1] < level))
        mids[j] = np.where(ok, (lo + hi) / 2, np.nan)

    with np.errstate(invalid="ignore"):
        pos = start - geom + np.where(np.any(np.isfinite(mids), 0),
                                      np.nanmean(mids, axis=0), np.nan)
    good = np.flatnonzero(np.isfinite(pos))
    if len(good) < len(pos) // 2:
        return None
    pos = np.interp(np.arange(len(pos)), good, pos[good])
    return pos - np.median(pos)


def curve(ink, floor):
    """A levels tool: crush everything paler than `floor`, restretch to 255.

    The move the README prices at 8.6 dB, applied on purpose so that one scan
    can be read with it and without it and nothing else differs. Monotone, so
    the axis is not supposed to care.
    """
    x = np.clip(ink.astype(np.float64) - floor, 0, None)
    top = x.max()
    return np.round(x * (255.0 / top if top else 1)).astype(np.uint8)


def read_both(ink, truth):
    """(r, per-lane r) for the centroid and for the axis, on the same pixels."""
    out = []
    for est in (None, lane_axis):
        song, sr, n = read_curves_page(ink, estimator=est)
        song, n, js, turned, rows = pilot_retime(song, sr, n)
        sr = (sr // rows) * rows
        joins = np.arange(1, round(len(song) / (sr // rows))) * (sr // rows)
        song = declick(highpass(song, BASELINE), joins, DECLICK)
        out.append(score(song, sr, truth))
    return out


def db(r):
    return 10 * np.log10(r * r / (1 - r * r))


def opt(argv, flag, cast=int):
    """--flag=a,b,c -> (a, b, c), and gone from argv."""
    got = [a.split("=", 1)[1] for a in argv if a.startswith(flag + "=")]
    return tuple(cast(v) for g in got for v in g.split(",")),         [a for a in argv if not a.startswith(flag + "=")]


def main(argv):
    global TOPS, WINDOW
    floors, argv = opt(argv, "--curve")
    tops, argv = opt(argv, "--tops")
    widths, argv = opt(argv, "--width", float)
    if tops:
        TOPS = tops
    if widths:
        WINDOW = widths[0]
    print(f"axis: window {WINDOW:g} px at pitch 38.7, ranks {TOPS} from the top")
    *scans, truth = argv
    print(f"{'scan':20s} {'tone curve':13s} {'centroid':>17s} {'axis':>17s}"
          f" {'axis - centroid':>16s}")
    for path in scans:
        raw = load_ink(path)
        for floor in (0,) + floors:
            ink = raw if not floor else curve(raw, floor)
            note = "raw" if not floor else f"crush <{floor}"
            if retouched(raw) and not floor:
                note = "already bent"
            (rc, lc), (ra, la) = read_both(ink, truth)
            print(f"{path:20s} {note:13s} "
                  f"{rc:7.4f} {db(rc):6.2f} dB {ra:7.4f} {db(ra):6.2f} dB"
                  f" {db(ra) - db(rc):+11.2f} dB")
            sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
