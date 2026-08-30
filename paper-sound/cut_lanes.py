#!/usr/bin/env python3
"""Cut a sheet of curves into one image per lane, into a folder.

paper_sound.py reads a sheet all the way back to audio. This stops one step
earlier and hands over the pictures instead: one lane is one second, so the
folder is the sheet's seconds, in order, as files.

The lanes are found the sheet's own way -- trim_paper() then find_tracks(),
which measures the page's own periodicity rather than looking for gaps, because
on a sheet of curves there need not be a gap: two neighbouring strokes can both
swing towards each other until the white between them closes. Nothing here
re-derives any of that, it only slices what find_tracks already decided.

Every cut is vertical and the strips are numbered left to right, which is the
order the seconds were printed in: 001 is the leftmost lane, and on a clocked
sheet that is the head clock, so 002 is the first second. Inside a strip time
runs down, a row to a sample, the way it does on the sheet. Nothing is cut
horizontally.

The defaults are set for a reader that CANNOT aim -- one that takes a
lane-wide centroid of the pixel value as it finds it, which is what picky.py
does and what most things will do. A strip is therefore not a plain crop: it
is masked down to a window riding its own stroke (APERTURE_MASK), trimmed to
its lane's own ink and stretched so every strip is a whole second, and written
as a negative (NEGATIVE) so that value and ink are the same thing. Measured on
04.png against the audio that went onto it, a plain crop reads 0.0042 and the
default strip 0.5790 -- paper_sound's own read of that sheet gets 0.4711.
`--aperture 0 --no-negative` gives the untouched crop back.

The one thing a sheet may not be able to tell you is which way up it lay. Two
clocks printed at different rates say it; two at the same rate -- a sheet
printed before they differed, as 04.png was -- cannot, and that is the one
fault that reports nothing and plays backwards. When the clocks come back
equal this says so, and --upside-down is the answer.

    cut_lanes.py sheet.png
    cut_lanes.py sheet.png -o lanes --bleed 4
    cut_lanes.py sheet.png --upside-down        # a sheet whose clocks cannot say
    cut_lanes.py sheet.png --aperture 0 --no-negative   # an untouched crop
    cut_lanes.py sheet_01.png sheet_02.png      # numbering runs on
    cut_lanes.py --selftest
"""
import argparse
import io
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

from paper_sound import (APERTURE, DPI, PILOT, PILOT_END, PITCH, STICKY,
                         STROKE, clock_order, lane_centroid, lane_drift,
                         lay_out, pilot_bin, pilot_lane, render_page,
                         trim_paper)
from read_tracks import find_tracks, ink_rows, load_ink, mend_png, odd_lane

# ------------------------------------------------------------------- defaults
# Every default lives here; the command line only overrides them, the same way
# paper_sound.py's own block does. Change a number here and the tool follows.

SUFFIX = "_lanes"       # folder made beside the scan: <scan><SUFFIX>/
PAD = 3                 # digits in a strip's name, so a plain sort is the
                        # playing order -- 002 before 010, which "2" is not
FORMAT = "tif"          # what a strip is written as: tif or png. TIFF because
                        # it is what a scanner hands over and what deflate below
                        # rides on; png is the same pixels in a container more
                        # things open without asking, and 12% larger here. The
                        # two are pixel for pixel identical -- the selftest
                        # cuts the same sheet both ways and checks it.
COMPRESS = "tiff_adobe_deflate"   # lossless, and the smallest of the TIFF
                        # codecs PIL offers here. On the 122 strips of 04.png,
                        # 33.4 MB raw goes to 6.65 against LZW's 7.83 and PNG's
                        # 7.47, for 0.16 s more across the whole sheet. Ignored
                        # when FORMAT is png, which brings its own deflate.
                        # Anything lossy would eat the one thing a strip
                        # carries, which is where the edge of the stroke fell
                        # between two pixels
APERTURE_MASK = APERTURE  # px of lane kept around the stroke, 0 = keep the
                        # whole crop. On, because without it a strip off real
                        # paper is not readable at all by anything that takes a
                        # LANE-WIDE centroid -- and picky.py takes a lane-wide
                        # one. A stroke fills a tenth of its lane, so nine
                        # tenths of what a wide window weighs is whatever else
                        # landed in that lane; paper_sound's own reader handles
                        # that by closing an APERTURE onto the stroke it just
                        # found. picky.py can now be told to do the same, with
                        # --aperture, and on a strip cut without a mask that
                        # is worth 0.6314 -> 0.9110 -- but not the whole of
                        # what the mask is worth, because a strip does not
                        # carry where the neighbouring lanes are, and on a
                        # 41 px pitch the neighbour's stroke swings into the
                        # bleed. Same sheet, same cut, mask alone: 0.9548.
                        # Doing it here, to the pixels, puts that read inside
                        # the file, and needs no reader to have the flag. Measured on 04.png against the
                        # audio that went onto it: 0.0042 off the plain crop,
                        # 0.5790 off the masked one, against 0.4711 for
                        # paper_sound's own read of the same sheet. It also
                        # trims each strip to its lane's own ink and stretches
                        # every lane to the median, because a strip that is not
                        # a whole second leaves anything concatenating them
                        # drifting out of time. Set to 0 for an untouched crop.
NEGATIVE = True         # write the strips as negatives, ink bright on black.
                        # On, because the reader these are for weighs pixel
                        # VALUE rather than ink: a reader that takes the
                        # centroid of the image as given has a paper-white
                        # strip tracking the paper, and the wave comes back
                        # mirrored -- exactly r = -1.000 against the audio that
                        # was printed. picky.py now inverts such a frame
                        # itself, so --no-negative no longer breaks it (0.3841
                        # paper-white against 0.6314, on an unmasked cut); the
                        # default stays on for every reader that does not.
                        # A mirrored mono wave is INAUDIBLE on its own, so this
                        # is the one default here that fixes nothing you can
                        # hear; it is on because the file is then right rather
                        # than inaudibly wrong, and the moment anything sums or
                        # compares two of these the sign is back to mattering.
                        # --no-negative for a strip that looks like the sheet.
                        # paper_sound's own reader inverts on load (load_ink)
                        # and does not care either way.
BLEED = STROKE / 2      # px kept on either side of the cut. find_tracks picks
                        # the offset whose cut lines carry the least ink, so a
                        # cut already lands in paper -- this is for the
                        # anti-alias fringe beside the stroke, not for the
                        # stroke. Half a stroke is small enough that a
                        # neighbour at rest (MARGIN px clear of the boundary by
                        # construction) does not appear in the strip.
                        # ponytail: a rectangle cannot follow a crooked
                        # lane -- but measured, the bleed is not what decides
                        # that. Strips off a sheet lying 17.6 px out over its
                        # height (the crooked sheet in paper_sound.py) read
                        # back at r 1.00000 against the audio that was printed,
                        # and widening the crop to the traced stroke changes
                        # them in no figure. What gives way first is the
                        # SEGMENTATION: past about 30 px of skew a lane wanders
                        # further than its own pitch, vertical cuts cannot tell
                        # it from its neighbour, and the strips come back at
                        # 0.37. Deskew the page before that; no bleed reaches
                        # it. The other ceiling is the medium's rather than the
                        # cut's -- at 0.9 of full swing a stroke at a sixth of
                        # the sample rate crosses 28 px in 7 rows, the row is a
                        # smear, and an 8 px window cannot follow: 0.11 to 0.57
                        # in the strips against 0.70 to 0.89 in paper_sound's
                        # own read of that page. Both readers see that one.


def strip_dpi(path):
    """The scan's own dpi, or the format's, so a strip prints at actual size.

    Through mend_png for the same reason load_ink is: one scanner here writes
    pHYs with a CRC it computed wrong, and pHYs is the very chunk this asks
    for. The ink went through the mend and the dpi did not, so sheetD_scan1
    and _2 -- two real scans sitting in this folder -- came apart here with
    PIL's own traceback before a strip was cut. load_ink says the file was
    mended; this one has nothing to add.
    """
    try:
        with Image.open(path) as im:
            d = im.info.get("dpi")
    except UnidentifiedImageError:
        fixed, _ = mend_png(path)
        if fixed is None:
            raise
        with Image.open(io.BytesIO(fixed)) as im:
            d = im.info.get("dpi")
    return tuple(float(v) for v in d) if d else (DPI, DPI)


def clock_at(ink, x0, x1, span, drift=0.0):
    """The printed rate of the clock in this lane, or None if it is not one.

    Read through lane_centroid, not through a lane-wide centroid of the crop.
    The cheap one is enough on a sheet straight out of render_page and is not
    enough on paper: on 04.png here it finds the head clock and misses the tail
    one, which is the difference between a sheet that says nothing about its
    orientation and one that is never asked.
    """
    # The window follows the pitch, as it does in read_curves_page: a wide
    # lane smears its stroke across more px of a row, and a clock read through
    # a window meant for 38.7 px is the one measurement whose failure turns a
    # sheet round.
    pos = lane_centroid(ink, x0, x1, span, drift, APERTURE * span / PITCH,
                        STICKY)
    got = pilot_bin(pos) if pos is not None else None
    return got and got[1:]


def clocks(ink, lanes):
    """The printed rates of the two end lanes, and the rows a sample got.

    -> ([rate or None, rate or None], [rows or None, rows or None]).


    A sheet is symmetric under a 180 degree turn in every respect but one: its
    two clocks are printed at different rates, PILOT on the left and PILOT_END
    on the right, so reading PILOT_END first is the one mark of a turned sheet.
    Two clocks at the SAME rate cannot say -- that sheet was printed before the
    rates differed -- and that is worth saying out loud rather than treating as
    a quiet no: it is the one case where a sheet plays backwards and nothing
    anywhere reports a fault. 04.png here is such a sheet, and reads 0.0042
    against the audio that went onto it the way it was scanned and 0.5790
    turned.
    """
    span = int(np.median([b - a for a, b in lanes]))
    # Read along the page's skew, the way every lane of a read is. The clock is
    # a tone crossing its lane 200-odd times, so a window walking off the
    # stroke costs the tone and not just precision: on sheetB_scan here,
    # straight down, only one end read as a clock, and one clock is not
    # evidence of which way up a sheet lay -- so the sheet came out backwards
    # in silence. It costs a first pass over the page (lane_drift, the reader's
    # own), which is the right price for the one measurement whose failure is
    # catastrophic.
    drift = lane_drift(ink, lanes, span, APERTURE * span / PITCH, STICKY)
    got = [clock_at(ink, *lanes[k], span, drift) for k in (0, -1)]
    return ([g and g[0] for g in got], [g and g[1] for g in got])


def stretch(a, n):
    """Stretch an image to exactly n rows, the way resample() stretches a lane."""
    src = np.linspace(0, len(a) - 1, n)
    i = np.minimum(np.floor(src).astype(int), len(a) - 2)
    f = (src - i)[:, None]
    return a[i] * (1 - f) + a[i + 1] * f


def traces(ink, lanes, narrow):
    """Where each lane's stroke actually runs: (rows, page column per row).

    The first pass of a read, and nothing more -- read_curves_page's own two
    steps, the skew from the mean of the lanes' centroids and then each lane
    ridden along it. What it is for here is aiming the aperture, not sampling.

    The skew is read_curves_page's own -- lane_drift(), the same call it
    makes -- so the two cannot disagree about how crooked the page is. The
    second pass is lane_centroid(..., absolute=True), which returns the trace
    already in the page's columns, so the frame that aims the mask is the
    frame that read the wav.
    """
    span = int(np.median([b - a for a, b in lanes]))
    drift = lane_drift(ink, lanes, span, narrow, STICKY)
    return [lane_centroid(ink, a, b, span, drift, narrow, STICKY, absolute=True)
            for a, b in lanes]


def cut(path, outdir, pitch=None, bleed=BLEED, turn=None, negative=NEGATIVE,
        aperture=None, fmt=FORMAT, first=1):
    """Write one TIFF per lane of `path`; returns how many.

    `turn` forces the 180 degree turn on or off; None measures it off the
    clocks. `aperture` masks each strip down to that many px around its own
    stroke; 0 leaves the crop alone, and None scales APERTURE_MASK by the
    sheet's own pitch the way the reader's window does -- mask a 77.4 px lane
    at the 38.7 px figure and the mask cuts into the stroke it was meant to
    keep.
    """
    ink = trim_paper(load_ink(path))
    lanes = find_tracks(ink, pitch)
    if not lanes:
        raise SystemExit(f"{path}: no lanes found")
    if aperture is None:
        aperture = APERTURE_MASK * np.median([b - a for a, b in lanes]) / PITCH

    speed = None            # rows a sample got, when the clocks were asked
    if turn is None:
        ends, rows_said = clocks(ink, lanes) if PILOT else ([None] * 2, [None] * 2)
        if rows_said[0] == rows_said[1]:
            speed = rows_said[0]
        # The rule itself is paper_sound's, called rather than copied: this
        # reader turns the IMAGE where that one turns the AUDIO, and the two
        # must not be free to drift apart about WHETHER a sheet was turned.
        turn, said = clock_order(ends, "cut")
        if turn:
            print(f"{path}: the clocks read back in the wrong order -- the "
                  f"sheet went on the glass upside down; the strips have been "
                  f"turned. The file on disk is untouched")
        elif said:
            print(f"{path}: {said}")
        elif ends.count(None) == 1:
            got = ends[0] if ends[0] is not None else ends[1]
            print(f"{path}: only one end reads as a clock ({got} samples a "
                  f"cycle, at the {'head' if ends[0] is not None else 'far'} "
                  f"end) -- the other may be damaged, or past where the grid "
                  f"stopped counting, or the lane may not be a clock at all. "
                  f"One end cannot say which way up this sheet lay, so it has "
                  f"been left alone. If it plays backwards, cut it again with "
                  f"--upside-down")
        elif PILOT and ends == [None, None]:
            # do_read says this too, and it is worth as much here: the lanes at
            # the ends are not clocks, so either the sheet was printed without
            # them or the grid is miscounted badly enough that the end lanes
            # are not the ends. The strips come out looking perfectly ordinary
            # either way.
            print(f"{path}: no clock lanes found -- printed without them, or "
                  f"the lane grid is miscounted. Cutting untimed, and this "
                  f"sheet cannot say which way up it lay")
        if 2 in rows_said:
            # The strips are pictures of ROWS, and this sheet put two rows in
            # every sample, so a strip is half a second and not a second. Said
            # rather than fixed: thinning the picture would break the one thing
            # a strip promises, that a row of it is a row of the paper. Read a
            # sheet like this with paper_sound, which divides by what the same
            # clocks say; a reader that takes a strip for a second plays it at
            # half speed and has nothing to warn it.
            print(f"{path}: the clocks say two rows to the sample -- a strip "
                  f"here is HALF a second, and its rows are the paper's rows. "
                  f"paper_sound reads this sheet at the right speed; anything "
                  f"that takes a strip for a second will not")
    if turn:
        # Segmented again rather than mirrored, and it is worth the second
        # pass: the grid is not symmetric about the middle of the page, so
        # mirroring a lane's column pair lands on a frame the search would
        # never have chosen -- the same ink, to the pixel, sitting 9 px along
        # inside a strip 9 px wider on the other side. Turning the array first
        # and cutting it as an upright sheet gives the strips the sheet would
        # have given had it gone on the glass the right way round, which is
        # the whole point of noticing.
        ink = ink[::-1, ::-1]
        lanes = find_tracks(ink, pitch)

    # The same warning read_curves_page prints, from the same place, because a
    # lane that is not one costs a strip here and a second there.
    said = odd_lane(lanes)
    if said:
        print(f"{path}: {said}")

    dpi = strip_dpi(path)
    b = int(round(bleed))
    tr = traces(ink, lanes, aperture) if aperture else None
    # One lane is one second, so a strip's height is the sample rate -- and
    # trimming each lane to its own ink leaves them 100 rows apart on a real
    # scan, which a reader that simply concatenates strips hears as the whole
    # sheet drifting out of time. read_curves_page settles it the same way, by
    # stretching every lane to the median; here the stretch is done to the
    # picture, so the file itself is a second.
    rate = None
    if tr is not None:
        heights = [r[1] - r[0] for r, _ in filter(None, tr)]
        if not heights:
            raise SystemExit(f"{path}: no lane on this sheet could be aimed at")
        rate = int(np.median(heights))
        if speed and speed > 1:
            # A whole number of SAMPLES, not just of rows. At two rows a sample
            # an odd strip is half a sample short, and a reader that
            # concatenates strips carries that half into the next lane: 59
            # samples across a 118-lane sheet, which is a tenth of a second by
            # the far edge. Measured on sheetD_scan2 through picky.py -- the
            # same read scores 0.04 with the drift in it and 0.88 with it taken
            # out, because correlation against the source is decided by single
            # samples of offset. The strips are stretched to a common height
            # anyway, so this costs one row of picture and nothing else.
            rate -= rate % speed
    for k, (x0, x1) in enumerate(lanes, first):
        lo, hi = max(0, x0 - b), min(ink.shape[1], x1 + b)
        crop = ink[:, lo:hi]
        if tr is not None and tr[k - first] is not None:
            (r0, r1), col = tr[k - first]
            x = np.arange(lo, hi)[None, :]
            crop = np.where(np.abs(x - col[:, None]) <= aperture / 2,
                            crop[r0:r1], 0)
            crop = np.round(stretch(crop, rate)).astype(np.uint8)
        elif tr is not None:
            # No trace to aim at, so no mask. The strip is still a second, and
            # a folder where one file is the page's height and the rest are the
            # rate drifts out of time from that lane on -- so it is trimmed and
            # stretched like the others. What it cannot be given is the mask,
            # and unmasked is the one state this tool exists to avoid: nine
            # tenths of what a lane-wide centroid weighs is then whatever else
            # landed in the lane (0.0042 against 0.5790, measured on 04.png).
            # Hence the noise about it rather than a quietly different file.
            rows = ink_rows(crop)
            print(f"{path}: strip {k:0{PAD}d} could not be aimed -- written "
                  f"unmasked, and a lane-wide centroid will read it as noise")
            crop = np.round(stretch(crop[rows[0]:rows[1]] if rows else crop,
                                    rate)).astype(np.uint8)
        # Back to paper-white: load_ink inverted the page to measure it, and
        # a strip should look like the sheet it came off unless asked not to.
        Image.fromarray(crop if negative else 255 - crop).save(
            os.path.join(outdir, f"{k:0{PAD}d}.{fmt}"), dpi=dpi,
            **({"compression": COMPRESS} if fmt == "tif" else {}))
    # Said here rather than by the caller, because this is where the sheet's
    # own speed was read. A strip is a picture of ROWS, and how many seconds
    # those rows are is the one thing about it that is not on its face.
    if rate is not None:
        if speed and speed > 1:
            # Two rates, both named, because they are not the same number and
            # naming one of them is how a sheet gets played at the wrong speed.
            # The AUDIO rate is the strip's height whatever the sheet's speed --
            # half as many samples, in half a second -- and it is what
            # paper_sound writes. The ROW rate is what a reader that takes one
            # row for one sample has to be given instead: picky.py and anything
            # like it, which cannot know that two rows here are one sample.
            print(f"  one strip is {rate} rows and holds 1/{speed} of a second,"
                  f" so the audio on it is {rate} Hz -- and a reader that takes"
                  f" one row for one sample has to play it at {rate * speed} Hz")
        else:
            print(f"  one lane is one second: {rate} rows, so {rate} Hz")
    return len(lanes)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scans", nargs="*", help="sheets to cut, in playing order")
    p.add_argument("-o", "--out",
                   help=f"folder for the strips (default: <first scan>{SUFFIX})")
    p.add_argument("--pitch", type=float,
                   help=f"lane pitch in px (default: measured; the format's own "
                        f"is {PITCH:g})")
    p.add_argument("--bleed", type=float, default=BLEED,
                   help=f"px kept either side of the cut (default {BLEED:g})")
    p.add_argument("--upside-down", action="store_true",
                   help="the sheet was scanned the wrong way round: turn it 180 "
                        "degrees before cutting. Sheets whose two clocks differ "
                        "are turned on their own; this is for one printed "
                        "without clocks, or before the clocks differed")
    p.add_argument("--format", choices=("tif", "png"), default=FORMAT,
                   help=f"container for the strips (default {FORMAT}); the "
                        f"pixels are the same either way")
    p.add_argument("--aperture", type=float, default=None,
                   help=f"mask each strip down to this many px around its own "
                        f"stroke, so a lane-wide centroid reads it the way "
                        f"paper_sound does; 0 is off. The default is "
                        f"{APERTURE_MASK:g} px scaled by the sheet's own pitch, "
                        f"as in the reader. It "
                        f"also trims each strip to its lane's own ink and "
                        f"stretches every lane to the median, so a strip is a "
                        f"whole second")
    p.add_argument("--negative", action="store_true", default=NEGATIVE,
                   help=f"write the strips as negatives, ink bright on black, "
                        f"for a reader that takes the centroid of pixel value "
                        f"rather than of ink -- picky.py does (default "
                        f"{'on' if NEGATIVE else 'off'})")
    p.add_argument("--no-negative", dest="negative", action="store_false",
                   help="write the strips paper-white, looking like the sheet")
    p.add_argument("--selftest", action="store_true",
                   help="print a sheet, cut it, check every strip")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.scans:
        raise SystemExit("give a sheet to cut, or --selftest")
    for path in args.scans:
        if not os.path.exists(path):
            raise SystemExit(f"{path}: no such file")

    outdir = args.out or args.scans[0].rsplit(".", 1)[0] + SUFFIX
    os.makedirs(outdir, exist_ok=True)
    # The strips are numbered from one every time, so a shorter sheet cut into
    # the folder a longer one left behind keeps the tail of the old cut -- in
    # order, correctly named, indistinguishable, and anything concatenating the
    # folder plays those seconds as part of this sheet. Only what this tool
    # writes is cleared: NNN.tif and NNN.png, nothing else in the folder.
    stale = [f for f in os.listdir(outdir)
             if len(f) == PAD + 4 and f[:PAD].isdigit() and f[PAD:] in (".tif", ".png")]
    for f in stale:
        os.remove(os.path.join(outdir, f))
    if stale:
        print(f"{outdir}/: {len(stale)} strip{'s' if len(stale) != 1 else ''} "
              f"from an earlier cut removed")
    n = 1
    for path in args.scans:
        try:
            cut_n = cut(path, outdir, args.pitch, args.bleed,
                        True if args.upside_down else None, args.negative,
                        args.aperture, args.format, n)
        except ValueError as e:
            # What read_curves_page does with the same sheet: a page with no
            # ink on it, or none that can be centred, is a thing to say rather
            # than a traceback. paper_sound.main wraps its own read this way.
            raise SystemExit(f"{path}: {e}")
        print(f"{path}: {cut_n} lanes -> {outdir}/"
              f"{n:0{PAD}d}.{args.format} .. "
              f"{n + cut_n - 1:0{PAD}d}.{args.format}")
        n += cut_n
    print(f"wrote {n - 1} strips to {outdir}/")


def selftest():
    """A clocked sheet out of paper_sound's own printer, cut both ways up, and
    every strip checked against the second that went into it.

    The centroid of a strip's ink, row by row, IS the sample -- that is the
    whole format -- so correlating it against the lane's own signal catches
    every way this can go wrong at once: a lane lost or invented, a strip
    numbered off by one, a stroke clipped by the cut, the page saved turned.
    Cutting the same sheet upside down has to land on the same strips, pixel
    for pixel, or the turn is not a turn.
    """
    import contextlib
    import io as _io
    import tempfile

    sr, n, pitch = 400, 20, PITCH
    # Half swing for the sheet that gets cut, because a clock lane next to a
    # neighbour swinging harder than that stops reading: at 0.9 the head clock
    # comes back None, the turn is not seen, and this would fail for a reason
    # that has nothing to do with cutting. The loud page is tested below
    # instead, where the only claim being made is the lane count.
    sig = 0.5 * np.concatenate([np.sin(2 * np.pi * 3 * (k + 1) * np.arange(sr) / sr)
                                for k in range(n)])
    # The clocks are what the turn is read from, so a sheet without them cannot
    # test it. Bracketing the field is do_print's own arrangement.
    whole = np.concatenate([pilot_lane(sr), sig, pilot_lane(sr, PILOT_END)])
    edges, _ = lay_out(whole, sr, n + 2, pitch, 0.0)
    width = int(np.ceil(max(r.max() for _, r in edges) + pitch))
    page = 255 - np.round(render_page(edges, pitch, width) * 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        strips = {}
        for way, laid in (("up", page), ("down", page[::-1, ::-1])):
            sheet = os.path.join(tmp, way + ".png")
            out = os.path.join(tmp, way)
            os.makedirs(out, exist_ok=True)
            Image.fromarray(laid).save(sheet, dpi=(DPI, DPI))
            got = cut(sheet, out)
            assert got == n + 2, f"{way}: cut {got} lanes out of a sheet of {n + 2}"
            strips[way] = [np.asarray(Image.open(os.path.join(out, f"{k:0{PAD}d}.{FORMAT}")))
                           for k in range(1, got + 1)]

        assert all(np.array_equal(a, b)
                   for a, b in zip(strips["up"], strips["down"])), \
            "the turned sheet did not cut to the same strips"

        # Read the way the strips are meant to be read, and the way this file's
        # defaults promise they can be: a LANE-WIDE centroid of the pixel value
        # as it sits in the file, which is all picky.py does. Inverting here
        # instead would test a reader nobody is going to use.
        def wave(a):
            a = np.asarray(a, float) if NEGATIVE else 255.0 - np.asarray(a, float)
            return (a * np.arange(a.shape[1])).sum(1) / np.maximum(a.sum(1), 1e-9)

        for k, want in ((0, PILOT), (-1, PILOT_END)):
            got = pilot_bin(wave(strips["up"][k]))
            assert got and got[1] == want, \
                f"strip {k} does not carry the {want}-sample clock"

        for k, a in enumerate(strips["up"][1:-1]):     # the clocks are not audio
            assert a.shape[0] == sr, f"strip {k + 2} is {a.shape[0]} rows, not {sr}"
            r = np.corrcoef(wave(a), sig[k * sr:(k + 1) * sr])[0, 1]
            assert r > 0.999, f"strip {k + 2} reads back at r={r:+.4f}"

        # A --rows 2 sheet, which is what every sheet on paper here is: the
        # clocks' period doubles in ROWS, so the same tone reads back as two
        # rows to the sample. The picture is not what this has to get right --
        # a strip is rows either way -- but the rates named under it are.
        odd = sr + 1        # an ODD lane height: half a sample short at speed 2
        sig2 = 0.5 * np.concatenate(
            [np.sin(2 * np.pi * 3 * (k + 1) * np.arange(odd) / odd)
             for k in range(n)])
        whole2 = np.concatenate([pilot_lane(odd, 2 * PILOT), sig2,
                                 pilot_lane(odd, 2 * PILOT_END)])
        edges2, _ = lay_out(whole2, odd, n + 2, pitch, 0.0)
        w2 = int(np.ceil(max(r.max() for _, r in edges2) + pitch))
        two = 255 - np.round(render_page(edges2, pitch, w2) * 255).astype(np.uint8)
        out2 = os.path.join(tmp, "two")
        os.makedirs(out2, exist_ok=True)
        Image.fromarray(two).save(os.path.join(tmp, "two.png"), dpi=(DPI, DPI))
        said = _io.StringIO()
        with contextlib.redirect_stdout(said):
            cut(os.path.join(tmp, "two.png"), out2)
        assert "two rows to the sample" in said.getvalue(), (
            f"a --rows 2 sheet was cut in silence: {said.getvalue().strip()!r}")
        # Whole samples, not whole rows: an odd strip at two rows a sample
        # walks half a sample into the next lane for whoever concatenates them.
        for k in range(1, n + 3):
            with Image.open(os.path.join(out2, f"{k:0{PAD}d}.{FORMAT}")) as im:
                assert im.size[1] % 2 == 0, (
                    f"strip {k} of a --rows 2 sheet is {im.size[1]} rows, "
                    f"which is half a sample short")
        assert f"the audio on it is {odd - 1} Hz" in said.getvalue(), (
            f"a --rows 2 sheet named the wrong audio rate: "
            f"{said.getvalue().strip()!r}")
        assert f"play it at {(odd - 1) * 2} Hz" in said.getvalue(), (
            f"a --rows 2 sheet did not name the row rate a strip reader needs: "
            f"{said.getvalue().strip()!r}")

        # The claim this file makes about its two containers, actually made.
        # It was printed for a long time on the strength of nobody having
        # written a png here at all.
        as_png = os.path.join(tmp, "png")
        os.makedirs(as_png, exist_ok=True)
        assert cut(os.path.join(tmp, "up.png"), as_png, fmt="png") == n + 2
        for k in range(1, n + 3):
            with Image.open(os.path.join(as_png, f"{k:0{PAD}d}.png")) as im:
                assert np.array_equal(np.asarray(im), strips["up"][k - 1]), \
                    f"strip {k} is not the same pixels in png as in tif"

        # A sheet whose pHYs carries a CRC its writer computed wrong. One
        # scanner here does that to every file it makes, and the ink was
        # mended for it while the dpi was not -- so this file came apart on
        # sheetD_scan1 and _2 with PIL's own traceback, before a strip was
        # cut. The strips have to come back the same as off the sound sheet.
        raw = bytearray(open(os.path.join(tmp, "up.png"), "rb").read())
        at = raw.find(b"pHYs")
        assert at > 0, "the test sheet was saved without a pHYs chunk to break"
        end = at + 4 + int.from_bytes(raw[at - 4:at], "big")
        raw[end:end + 4] = bytes(b ^ 0xff for b in raw[end:end + 4])
        bad = os.path.join(tmp, "badcrc.png")
        open(bad, "wb").write(bytes(raw))
        out = os.path.join(tmp, "badcrc_lanes")
        os.makedirs(out, exist_ok=True)
        said = _io.StringIO()
        with contextlib.redirect_stdout(said):
            got = cut(bad, out)
        assert got == n + 2, f"the mended sheet cut into {got} lanes"
        assert "mended in memory" in said.getvalue(), (
            f"a mended sheet was cut without a word: {said.getvalue().strip()!r}")
        for k in range(1, got + 1):
            with Image.open(os.path.join(out, f"{k:0{PAD}d}.{FORMAT}")) as im:
                assert np.array_equal(np.asarray(im), strips["up"][k - 1]),                     f"strip {k} off the mended sheet is not the same pixels"

        # A crooked sheet, which the square one above cannot stand in for: the
        # cut is a rectangle and the stroke is not vertical inside it. 17.6 px
        # of skew over the sheet is what the crooked scan in paper_sound.py
        # measured, and it costs nothing -- but only once the sheet's own tilt
        # is out of the way. A strip CARRIES that tilt, as a ramp down its
        # centroid; it is the paper rather than the cut, picky.py takes it off
        # with its adjustment 2 or 3 and the wav side with BASELINE, and this
        # takes it off both sides before comparing. Scored against the raw
        # signal instead, an honest strip off a crooked sheet reads 0.73.
        bent_edges, _ = lay_out(whole, sr, n + 2, pitch, 17.6 / sr)
        w = int(np.ceil(max(r.max() for _, r in bent_edges) + pitch))
        bent = os.path.join(tmp, "bent.png")
        Image.fromarray(255 - np.round(render_page(bent_edges, pitch, w) * 255)
                        .astype(np.uint8)).save(bent, dpi=(DPI, DPI))
        out = os.path.join(tmp, "bent_lanes")
        os.makedirs(out, exist_ok=True)
        got = cut(bent, out)
        assert got == n + 2, f"the crooked sheet cut into {got} lanes"

        def flat(v):
            y = np.arange(len(v))
            return v - np.polyval(np.polyfit(y, v, 1), y)

        for k in range(2, n + 2):
            with Image.open(os.path.join(out, f"{k:0{PAD}d}.{FORMAT}")) as im:
                v = flat(wave(np.asarray(im)))
            r = np.corrcoef(v, flat(sig[(k - 2) * sr:(k - 1) * sr]))[0, 1]
            assert r > 0.999, f"crooked strip {k} reads back at r={r:+.4f}"

        # A loud sheet through cut() itself. The lane COUNT at 0.7 and 0.9 is
        # checked further down on find_tracks alone, and the count is the easy
        # half: the mask has to stay on a stroke swinging the width of its own
        # lane, and the trim and the stretch have to leave the strip a second.
        # Clockless, because a clock beside neighbours swinging this hard stops
        # reading -- which is why the sheet cut above is a quiet one. Measured:
        # median 1.00000, with one strip in the middle of the sheet at 0.938
        # where the neighbour's stroke comes inside the window.
        loud = 0.9 * np.concatenate([np.sin(2 * np.pi * 3 * (k + 1) * np.arange(sr) / sr)
                                     for k in range(30)])
        loud_edges, _ = lay_out(loud, sr, 30, pitch, 0.0)
        w = int(np.ceil(max(e.max() for _, e in loud_edges) + pitch))
        shout = os.path.join(tmp, "loud.png")
        Image.fromarray(255 - np.round(render_page(loud_edges, pitch, w) * 255)
                        .astype(np.uint8)).save(shout, dpi=(DPI, DPI))
        out = os.path.join(tmp, "loud_lanes")
        os.makedirs(out, exist_ok=True)
        with contextlib.redirect_stdout(_io.StringIO()):   # no clocks to find
            got = cut(shout, out)
        assert got == 30, f"the loud sheet cut into {got} lanes, not 30"
        rs = []
        for k in range(1, 31):
            with Image.open(os.path.join(out, f"{k:0{PAD}d}.{FORMAT}")) as im:
                a = np.asarray(im)
            assert a.shape[0] == sr, f"loud strip {k} is {a.shape[0]} rows, not {sr}"
            rs.append(np.corrcoef(wave(a), loud[(k - 1) * sr:k * sr])[0, 1])
        assert np.median(rs) > 0.999 and min(rs) > 0.9, (
            f"loud strips read back at median {np.median(rs):.4f}, "
            f"worst {min(rs):.4f}")

        # And the sheet this CANNOT cut, kept as a test of what it says rather
        # than of what it does. At full swing there is no paper left between
        # the lanes for a cut line to fall in -- MARGIN is 1.5 px of it -- so
        # every offset costs the same, the grid lands half a pitch out, and the
        # lanes at both ends come back as halves of one. That is a known
        # ceiling, and not worth laying the grid a different way for a sheet
        # whose every second runs at full excursion: prep.bat leaves a crest of
        # 5 to 7.5, which is an rms near a fifth of this. What must not happen
        # is that it stops saying so.
        full = np.concatenate([np.sin(2 * np.pi * 3 * (k + 1) * np.arange(sr) / sr)
                               for k in range(30)])
        full_edges, _ = lay_out(full, sr, 30, pitch, 0.0)
        w = int(np.ceil(max(e.max() for _, e in full_edges) + pitch))
        Image.fromarray(255 - np.round(render_page(full_edges, pitch, w) * 255)
                        .astype(np.uint8)).save(shout, dpi=(DPI, DPI))
        said = _io.StringIO()
        with contextlib.redirect_stdout(said):
            cut(shout, out)
        assert "segmented as a lane that is not one" in said.getvalue(), (
            f"a sheet at full swing was cut into halves in silence: "
            f"{said.getvalue().strip()!r}")

        # Two sheets into one folder number on rather than start again, which
        # is the whole of `first` and the only thing standing between a second
        # sheet and the first one's strips.
        run = os.path.join(tmp, "run")
        os.makedirs(run, exist_ok=True)
        one = cut(os.path.join(tmp, "up.png"), run)
        two = cut(bent, run, first=one + 1)
        assert sorted(os.listdir(run)) == [f"{k:0{PAD}d}.{FORMAT}"
                                           for k in range(1, one + two + 1)], \
            "a second sheet did not number on from the first"
    # The one thing a quiet sheet cannot test. find_pitch measures the page as
    # a grating, and every lane holds a different second, so a loud page is a
    # grating whose bars move: past about half the lane's swing in rms the
    # fundamental is beaten down and the peak lands on twice or three times the
    # pitch. The page is then cut into twice or three times too many lanes and
    # every second after the first is shifted, in silence. This same sheet came
    # back as 40 lanes before find_pitch learned to ask a half and a third of
    # its winner, and as 60 at full swing.
    # Asked at three sizes, because the fault is not a property of the swing
    # alone and a guard measured on one size of page is not measured: at 0.9 of
    # full swing this sheet's own 20 lanes came back right while 30 came back
    # as 22 and 120 as 240, all three from the same code.
    for lanes in (n, 30, 120):
        for amp in (0.7, 0.9):
            loud = amp * np.concatenate([np.sin(2 * np.pi * 3 * (k + 1) * np.arange(sr) / sr)
                                         for k in range(lanes)])
            e, _ = lay_out(loud, sr, lanes, pitch, 0.0)
            w = int(np.ceil(max(r.max() for _, r in e) + pitch))
            got = len(find_tracks(trim_paper(
                np.round(render_page(e, pitch, w) * 255).astype(np.uint8))))
            assert got == lanes, (f"a page of {lanes} lanes at {amp} of full "
                                  f"swing cut into {got}")

    print(f"selftest: {n + 2} lanes cut both ways up, clocks in order, "
          f"lane-wide centroid r > 0.999 on every strip; tif and png identical; "
          f"a broken pHYs CRC mends to the same strips; "
          f"a sheet 17.6 px crooked cuts to r > 0.999 as well; a loud one cuts "
          f"to strips and a full-swing one still says why it cannot; two sheets "
          f"number on; a --rows 2 sheet names both its rates; "
          f"pages of {n}, 30 and 120 lanes at 0.7 and 0.9 of full swing "
          f"still cut into as many")


if __name__ == "__main__":
    main()
