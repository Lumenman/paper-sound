#!/usr/bin/env python3
"""Print a strip of gratings, scan it, and let the paper say how fine a grating
it still keeps.

The format prints one stroke per lane and reads its position. The stroke has
two edges and they move together, which is why it beats every variable-area
layout drawn on the same paper -- see LAB 24. The next question that table
raises is whether a lane can carry MORE than two edges. A grating of N bars
shifted bodily by the sample has 2N edges, every one of them travelling the
full excursion, and the bound on a shift estimate goes as the sum of the
squared derivatives of the image, so N bars are N times the information --
until print and scan stop resolving them.

Where that stops is not knowable from a lens formula. A laser printer's dot
gain is not a convolution and a flatbed's MTF is not its dpi. So this prints
the question instead: one strip, one block per grating period, and the paper
answers.

    grating.py print -o strip.png
    grating.py read strip.png
    grating.py selftest

`read` measures two numbers per block:

  contrast  the fundamental's amplitude against the same grating drawn
            perfectly -- the MTF at that period. Diagnostic: it says WHY a
            period failed, and it is what a formula would have guessed at.
  sigma     how far the grating's measured position wanders down the strip,
            in pixels, measured as the difference between neighbouring
            lanes. The bars are straight by construction, so all of this is
            noise -- and it is the number the format actually spends, because
            one row is one sample and a sample's error is this. Differenced
            rather than absolute because the scanner walks the whole strip at
            once, and that walk is bigger than what is being measured: see
            do_read().

Both are measured in windows one PITCH wide, because a lane is all a grating
would get. Reading a grating across a whole block would average eleven lanes'
worth of ink and flatter it by 10 dB.

The reference block is the current format at rest: bars STROKE wide, PITCH
apart, which is a sheet of silence. Its sigma is this format's own noise floor
measured on the same paper in the same pass, so the last column -- the ratio of
the two -- is the answer with the printer and the scanner divided out. The
ratio is in dB of the project's own metric because both formats would be given
the same excursion: a grating's phase is unwrapped down the lane, so it swings
as far as a stroke does.

What this strip does NOT settle: cycle slips. A grating that loses a cycle
loses it for the rest of the second, and nothing here is modulated, so nothing
here can slip. That is the next sheet, not this one.
"""
import argparse

import numpy as np

from paper_sound import (APERTURE, BASELINE, DPI, MARGIN_MM, PAPER_DEFAULT,
                         PITCH, STROKE, box1d, mm_px, paper_mm, render_page,
                         sheet_px, trim_paper)
from read_tracks import load_ink

# Periods in px at 600 dpi: 19.7 px is 0.83 mm, 2.9 px is 0.12 mm. The printed
# stroke measures 4.32-4.59 px FWHM against 4.0 drawn (LAB 23.11), so the blur
# is about 1.9 px FWHM and the interesting ground is 3 to 8. The coarse end is
# there to have something that MUST survive: if 19.7 px comes back soft, the
# scan is wrong rather than the idea.
#
# None of them is a whole number of pixels, and that is the point. At a period
# of exactly 4 every bar in the block lands on the pixel grid the same way, so
# the block measures one piece of luck: drawn and never printed, period 6 reads
# 1.097 of its own ceiling and period 4 reads 0.936. Off the grid the bars walk
# through every sub-pixel phase and the block averages them -- the same nine
# periods then read 1.0000 +- 0.0015, which is the rasteriser divided out
# before the paper is asked anything. A grating in a real lane sits at whatever
# phase it sits at, so this is also the more honest question.
LAMBDAS = (19.7, 13.9, 9.9, 7.9, 6.1, 4.9, 4.1, 3.4, 2.9)

# Solid at both ends on purpose: trim_paper() cuts a page to its ink, so a
# block that reaches the very edge of the strip is what makes the block grid
# land where it was printed. Blank sits next to the last solid, for the paper's
# own level and noise -- in the middle of the strip rather than at its edge,
# where it would simply be trimmed off.
BLOCKS = ("solid", "stroke", *LAMBDAS, "blank", "solid")


def named(spec):
    return spec if isinstance(spec, str) else f"{spec:g}"


def block_bars(x0, w, spec):
    """(left, right) column pairs for one block: a grating, a stroke, or a fill."""
    if spec == "solid":
        return [(x0, x0 + w)]
    if spec == "blank":
        return []
    # The reference block is a lane of silence at the format's own geometry;
    # every other block is a 50% duty grating, which is how a shifted grating
    # would be drawn.
    period, bar = (PITCH, STROKE) if spec == "stroke" else (spec, spec / 2)
    n = int(w // period)
    off = x0 + (w - n * period) / 2
    return [(off + k * period, off + k * period + bar) for k in range(n)]


def ideal_amp(spec, solid):
    """Fundamental this block would have if print and scan cost nothing.

    A bar `b` wide at period `p` has a fundamental of 2/pi * sin(pi*b/p) of the
    ink it is drawn in -- and then the rasteriser samples it with a pixel,
    which is a box one pixel wide and costs another sinc(1/p). That second
    factor is ours, not the paper's, so it belongs in the ceiling: without it
    the finest block would be marked down 18% for the way `print` had to draw
    it. The scan's own pixel is NOT divided out, because that one is a real
    loss the format would take.
    """
    period, bar = (PITCH, STROKE) if spec == "stroke" else (spec, spec / 2)
    return 2 * solid / np.pi * np.sin(np.pi * bar / period) * np.sinc(1.0 / period)


def do_print(args):
    from PIL import Image

    width, _ = sheet_px(args.paper, args.dpi, args.margin_mm)
    rows = mm_px(args.mm, args.dpi)
    bw = width / len(BLOCKS)
    pairs = []
    for i, spec in enumerate(BLOCKS):
        pairs += block_bars(i * bw, bw, spec)
    edges = [(np.full(rows, a), np.full(rows, b)) for a, b in pairs]
    strip = render_page(edges, 0.0, width)

    paper_w, paper_h = paper_mm(args.paper)
    sheet_w, sheet_h = mm_px(paper_w, args.dpi), mm_px(paper_h, args.dpi)
    page = np.zeros((sheet_h, sheet_w), np.float32)
    if getattr(args, "cross", False):
        # The same strip twice, the second one transposed, on ONE sheet. LAB
        # 24.6 measured this paper's blur off vertical bars, which is the only
        # direction vertical bars can measure -- and a flatbed's carriage moves
        # the other way, so there is no reason for the two to agree. Two sheets
        # would answer it too and would answer it wrong: the difference between
        # two passes of the glass is the size of the thing being measured. On
        # one sheet in one pass it divides out.
        gap = mm_px(10, args.dpi)
        tall = strip.T
        h = rows + gap + tall.shape[0]
        y = (sheet_h - h) // 2
        x = (sheet_w - width) // 2
        page[y:y + rows, x:x + width] = strip
        page[y + rows + gap:y + h, x:x + tall.shape[1]] = tall
    else:
        y, x = (sheet_h - rows) // 2, (sheet_w - width) // 2
        page[y:y + rows, x:x + width] = strip
    out = args.out or ("cross.png" if getattr(args, "cross", False)
                       else "strip.png")
    Image.fromarray(255 - np.round(page * 255).astype(np.uint8)).save(
        out, dpi=(args.dpi, args.dpi))
    print(f"{out}: {len(BLOCKS)} blocks of {bw:.0f} px, ink {width}x{rows} px "
          f"= {width * 25.4 / args.dpi:.0f}x{args.mm:g} mm at {args.dpi} dpi"
          f"{', twice, the second one turned' if getattr(args, 'cross', False) else ''}")
    print("  " + "  ".join(named(s) for s in BLOCKS))
    print(f"  print at 100%, scan at {args.dpi} dpi greyscale with every "
          f"adjustment off -- the same rules as a sheet, and for the same "
          f"reason: this measures edges")


def sweep(crop, lams):
    """Mean per-row fundamental magnitude of `crop` at each trial period."""
    x = np.arange(crop.shape[1])
    win = np.hanning(crop.shape[1] + 2)[1:-1]
    k = 2 * np.pi / np.asarray(lams, float)[:, None]
    c = (np.cos(k * x) * win).T
    s = (np.sin(k * x) * win).T
    return np.hypot(crop @ c, crop @ s).mean(0)


def refine(crop, lam, span=0.03, steps=61):
    """The period near `lam` this crop actually carries."""
    lams = lam * np.linspace(1 - span, 1 + span, steps)
    return float(lams[int(np.argmax(sweep(crop, lams)))])


def per_row(crop, lam):
    """Amplitude and position, row by row, of the grating at period `lam`.

    The position is the fundamental's phase, unwrapped down the strip: skew and
    the page's own bend walk the phase past +-pi on a fine grating, and a
    wrapped phase reads as a step the size of a whole period.
    """
    x = np.arange(crop.shape[1])
    win = np.hanning(crop.shape[1] + 2)[1:-1]
    k = 2 * np.pi / lam
    a = crop @ (np.cos(k * x) * win)
    b = crop @ (np.sin(k * x) * win)
    amp = 2 * np.hypot(a, b) / win.sum()
    return amp, np.unwrap(np.arctan2(b, a)) * lam / (2 * np.pi)


def stroke_row(crop, narrow):
    """Centroid of the one stroke in this window, row by row.

    The same narrow window the reader closes onto a stroke, aimed at the
    block's own mean peak. A printed stroke does not move, so the aim here is
    perfect and the reader's aim is not -- which flatters the reference block,
    and so flatters the format this strip is trying to beat. That is the
    direction to be wrong in.
    """
    prof = crop.mean(0)
    c = int(prof.argmax())
    half = int(round(narrow / 2))
    lo, hi = max(0, c - half), min(crop.shape[1], c + half + 1)
    seg = crop[:, lo:hi]
    den = seg.sum(1)
    return np.where(den > 0, seg @ np.arange(lo, hi) / np.where(den > 0, den, 1),
                    np.nan)


def scatter(pos, win=BASELINE):
    """Rms wander of a position trace once the page's own bend is removed.

    The high pass is the reader's own BASELINE, so what is left is the noise
    the format would actually hear. The variance a moving average of `win`
    takes with it is 1/win, and it is put back rather than reported as quiet.
    """
    win = min(win | 1, max(3, len(pos) // 4 | 1))
    d = np.asarray(pos, float) - box1d(np.asarray(pos, float), win, 0)
    d = d[win:-win]
    good = d[np.isfinite(d)]
    if len(good) < 16:
        return float("nan")
    return float(good.std() / np.sqrt(1 - 1.0 / win))


def middle(crop):
    return crop[:, crop.shape[1] // 4:crop.shape[1] * 3 // 4]


def measure(ink, label):
    """One strip of blocks -> its table, and the rows behind it."""
    rows, width = ink.shape
    bw = width / len(BLOCKS)
    cuts = [ink[:, int(round(i * bw)):int(round((i + 1) * bw))]
            for i in range(len(BLOCKS))]

    solid = float(np.mean([middle(cuts[i]).mean()
                           for i, s in enumerate(BLOCKS) if s == "solid"]))
    blank = middle(cuts[BLOCKS.index("blank")])
    if solid <= 0:
        raise SystemExit(f"{label}: no ink -- is this the strip?")
    # The blank block is bare paper and the solid blocks are full ink, so the
    # one has to read far lighter than the other. When it does not, the block
    # grid is not on the strip and every number below is arithmetic on the
    # wrong pixels -- which is what a scan came back as in LAB 24.10, where the
    # flatbed left a 50 px band of its own down the right edge of the page.
    # That band is full height, so trim_paper() keeps it as a column of ink,
    # and the strip is then measured 100 px wider than it is. Caught here
    # rather than left to be read as a result: the first read of that scan
    # reported paper at 203 of 255 against ink at 18 and printed a table.
    if blank.mean() > solid / 4:
        raise SystemExit(
            f"{label}: the blank block reads {blank.mean():.0f} of ink against "
            f"{solid:.0f} in the solid ones, so the block grid is not on the "
            f"strip. Something that is neither paper nor strip survived the "
            f"trim -- a scanner's edge band is the usual one. Crop the scan to "
            f"the sheet and read it again")

    # The strip prints its own ruler: the reference block's period IS the
    # format's PITCH, so what it measures as is the scan's scale. Nothing here
    # has to be told the dpi it was printed at, or the one it came back at.
    ref = cuts[BLOCKS.index("stroke")]
    pitch_px = refine(ref, PITCH, span=0.06, steps=241)
    scale = pitch_px / PITCH
    narrow = APERTURE * scale

    print(f"{label}: {width}x{rows} px of ink, {len(BLOCKS)} blocks of "
          f"{bw:.0f} px, scale {scale:.4f}")
    print(f"  ink {solid:.0f} solid, paper {blank.mean():.1f} "
          f"+-{blank.std():.1f} of 255")
    print(f"  {'block':>8} {'period':>8} {'contrast':>9} {'sigma px':>9} "
          f"{'vs stroke':>10}")

    out = []
    for i, spec in enumerate(BLOCKS):
        if spec in ("solid", "blank"):
            continue
        crop = cuts[i]
        lam = pitch_px if spec == "stroke" else refine(crop, spec * scale)
        # Contrast off the WHOLE block, sigma off lane-wide windows. Contrast
        # is a property of the printer and the glass, so it is measured where
        # there is most of it to measure; a lane-wide window holds one period
        # of the reference block, and a Hann window one period wide reports a
        # sixth of the amplitude that is there.
        contrast = per_row(crop, lam)[0].mean() / ideal_amp(spec, solid)
        nw = max(1, int(crop.shape[1] // pitch_px))
        off = int((crop.shape[1] - nw * pitch_px) / 2)
        traces = []
        for k in range(nw):
            w = crop[:, off + int(k * pitch_px):off + int((k + 1) * pitch_px)]
            traces.append(stroke_row(w, narrow) if spec == "stroke"
                          else per_row(w, lam)[1])
        # Adjacent lanes, differenced. A lane's own trace is not this block's
        # noise: the scanner's transport and the page's bend walk every lane on
        # the strip the same way at once, and the high pass in scatter() takes
        # out the slow part of that and leaves the rest. Measured on the scan
        # of 04.09, blocks 4500 px apart carrying periods seven times apart
        # still track each other at 0.56 to 0.96, and their common walk is
        # 0.104 px of a 0.11-0.13 px sigma. An absolute sigma is then mostly
        # that shared walk -- identical in the reference block too, so the dB
        # column divides one scanner by the same scanner and reads +2.5 dB
        # where the paper is doing 8. Two neighbouring lanes see one page
        # wander and two independent noises; their difference keeps only the
        # noise, and /sqrt(2) is because there are two of them. The estimator
        # is the same on both sides of the ratio, so what is left is the
        # format.
        sigmas = [scatter(b - a) / np.sqrt(2)
                  for a, b in zip(traces, traces[1:])] or [scatter(traces[0])]
        # Median over the pairs, not the mean: one window that caught a hair or
        # the edge of a block spoils the two differences it sits in, and two
        # differences do not get to set the number for a whole period.
        out.append((spec, contrast, lam / scale, float(np.median(sigmas))))

    # A strip that never left the computer has nothing to wander: its bars are
    # straight to the last bit and every sigma is 1e-14. The dB column would
    # then be a ratio of two roundings, and it would read +65 dB and mean
    # nothing. The floor is a tenth of the finest thing a real scan has shown.
    ref_sigma = out[0][3]
    paper = ref_sigma > 0.0005
    for spec, contrast, lam, sigma in out:
        last = ("reference" if spec == "stroke" else
                f"{20 * np.log10(ref_sigma / sigma):+.2f} dB" if paper else "--")
        print(f"  {named(spec):>8} {lam:8.2f} {contrast:9.3f} {sigma:9.4f} "
              f"{last:>10}")

    if not paper:
        print(f"  the stroke stands still to {ref_sigma:.1e} px, so this strip "
              f"has never been printed. Contrast is the rasteriser and nothing "
              f"else; the dB column needs paper")
        return out

    fine = [r for r in out if r[0] != "stroke" and r[3] > 0]
    if not fine:
        return out
    spec, contrast, lam, sigma = min(fine, key=lambda r: r[3])
    swing = 2 * ((PITCH - STROKE) / 2 - 1.5)
    print(f"  best: {spec:g} px, {20 * np.log10(ref_sigma / sigma):+.2f} dB "
          f"against the stroke this same sheet prints")
    print(f"  a lane holds {PITCH / spec:.1f} cycles of it, and the phase wraps "
          f"{swing / spec:.1f} times across the format's {swing:.1f} px of "
          f"excursion -- so the sample has to move less than {spec / 2:.2f} px "
          f"a row for the unwrap to hold. It moves about 0.2")
    return out


def split_cross(ink, frac=0.02, least=8):
    """A cross sheet cut into its two strips: (across the page, down the page).

    Split on the FIRST band of rows carrying no ink, not the longest one. The
    across strip makes no such band -- its blank BLOCK is a column of paper and
    leaves every row of the strip still inked somewhere -- but the down strip
    is that same strip transposed, so its blank block IS a band of blank rows,
    a block wide. On A4 that is 363 rows against the 236 of the gap between the
    strips, and the longest run is then the wrong one: the split lands inside
    the second strip and hands back the last solid block as a whole strip. The
    selftest used to print an 80x120 sheet, whose block is narrower than the
    gap, which is why it never said so; it now prints one wider than 140 mm,
    where a block is the wider of the two.

    Blank is counted in strong pixels rather than summed intensity, the way
    trim_paper() counts it: bare paper scans at 8-10 of 255 of ink, and across
    4600 columns that sums to 9% of an inked row -- over any threshold low
    enough to still find a band.
    """
    floor = np.median(ink[::16, ::16])
    prof = (ink > (float(ink.max()) + floor) / 2).sum(1)
    off = prof < prof.max() * frac
    run = 0
    for i, v in enumerate(list(off) + [False]):
        if v:
            run += 1
        elif run >= least:
            return trim_paper(ink[:i - run]), trim_paper(ink[i:])
        else:
            run = 0
    raise SystemExit("no blank band across this scan -- is it a --cross "
                     "sheet? A single strip is read without --cross")


def do_read(args):
    ink = trim_paper(load_ink(args.scan)).astype(np.float64)
    if not args.cross:
        measure(ink, args.scan)
        return

    across, down = split_cross(ink)
    # Transposed, not rotated: `print --cross` lays the second strip down with
    # a plain transpose, so this is exactly its inverse and the blocks come
    # back in the order they were drawn in. A rotation would hand them back
    # mirrored, and the block grid is positional.
    a = measure(across, f"{args.scan} across the page")
    print()
    d = measure(down.T, f"{args.scan} down the page")

    # The one number this sheet exists for. Both strips are the same ink on the
    # same paper in the same pass, so what is left when one contrast is divided
    # by the other is the scanner's own asymmetry and nothing else -- no dpi,
    # no exposure, no second scan to disagree with the first.
    print()
    print(f"  {'block':>8} {'across':>9} {'down':>9} {'down/across':>12}")
    for (spec, ca, _, _), (_, cd, _, _) in zip(a, d):
        print(f"  {named(spec):>8} {ca:9.3f} {cd:9.3f} {cd / ca:12.3f}"
              if ca else "")
    fine = [(s_, ca, cd) for (s_, ca, _, _), (_, cd, _, _) in zip(a, d)
            if s_ != "stroke" and ca > 0.02]
    if not fine:
        return
    worst = min(fine, key=lambda r: r[2] / r[1])
    print(f"  the glass is {'' if worst[2] < worst[1] else 'NOT '}softer down "
          f"the page: at {worst[0]:g} px it keeps {worst[2] / worst[1]:.2f} of "
          f"what it keeps across")
    print("  slip.py needs this: its vernier beats the stroke by 13 dB when "
          "the blur down the page is small and loses by 8 when it is not")


def selftest():
    """The strip through the whole path: draw it, blur it, dirty it, read it."""
    import contextlib
    import io as _io
    import os
    import tempfile

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "strip.png")
        args = argparse.Namespace(paper="120x40", dpi=600, margin_mm=5.0,
                                  mm=15.0, out=png)
        with contextlib.redirect_stdout(_io.StringIO()):
            do_print(args)

        ink = trim_paper(load_ink(png)).astype(np.float64)
        assert ink.shape == (mm_px(15, 600), mm_px(110, 600)), (
            f"ink is {ink.shape}, want {(mm_px(15, 600), mm_px(110, 600))}")

        # Drawn and never printed: every grating must come back at full
        # contrast, at the period it was drawn at, and standing still. Anything
        # else here is this file's own arithmetic being wrong, before any paper
        # is blamed for it.
        bw = ink.shape[1] / len(BLOCKS)
        solid = middle(ink[:, :int(bw)]).mean()
        for spec in (19.7, 6.1, 4.1, 2.9):
            i = BLOCKS.index(spec)
            crop = ink[:, int(round(i * bw)):int(round((i + 1) * bw))]
            lam = refine(crop, spec)
            assert abs(lam - spec) < 0.02 * spec, f"{spec} px reads {lam:.2f}"
            amp, pos = per_row(crop, lam)
            # Against ideal_amp, which already carries the pixel the strip was
            # drawn with: what is left has to be 1.00, or the ceiling the paper
            # is being measured against is the wrong ceiling.
            drawn = amp.mean() / ideal_amp(spec, solid)
            assert abs(drawn - 1) < 0.01, f"{spec} px draws at {drawn:.4f} of its ceiling"
            assert scatter(pos) < 0.01, f"{spec} px wanders {scatter(pos):.4f} px"

        # A shift the strip never had: move one block bodily by a known number
        # of pixels and check the phase reports it. This is the whole claim the
        # strip is built on -- that a grating's phase IS a position -- and it is
        # the one thing a strip of straight bars cannot say about itself.
        i = BLOCKS.index(6.1)
        crop = ink[:, int(round(i * bw)):int(round((i + 1) * bw))]
        _, a = per_row(crop, 6.1)
        _, b = per_row(np.roll(crop, 3, axis=1), 6.1)
        moved = float(np.median(b - a)) % 6.1
        assert min(abs(moved - 3.0), abs(moved - 9.1)) < 0.05, (
            f"a 3 px shift reads as {moved:.3f} px")

        # Common mode: walk every row of the strip bodily by the same amount,
        # which is what a scanner's transport does. Every lane of a block then
        # carries the whole walk, so an absolute sigma reads it as noise and
        # the differenced one must not see it at all. This is the claim
        # do_read() rests on, and the one an absolute sigma gets wrong on
        # paper.
        walk = rng.integers(-1, 2, ink.shape[0])
        cols = (np.arange(ink.shape[1]) - walk[:, None]) % ink.shape[1]
        crop = ink[np.arange(ink.shape[0])[:, None], cols][
            :, int(round(i * bw)):int(round((i + 1) * bw))]
        n = int(crop.shape[1] // PITCH)
        tr = [per_row(crop[:, int(k * PITCH):int((k + 1) * PITCH)], 6.1)[1]
              for k in range(n)]
        assert scatter(tr[0]) > 0.5, f"the walk did not take: {scatter(tr[0]):.3f} px"
        d = float(np.median([scatter(b - a) / np.sqrt(2)
                             for a, b in zip(tr, tr[1:])]))
        assert d < 0.02, f"a common walk leaks {d:.4f} px into the difference"

        # And through a scan: blur, noise, and a page that is not flat. The
        # bend has to come out in the high pass and the noise has to stay in,
        # so sigma must rise off the floor and still sit well under a pixel.
        blur = np.ones(5) / 5
        dirty = np.apply_along_axis(np.convolve, 1, ink, blur, "same")
        dirty += rng.normal(0, 6, dirty.shape)
        crop = dirty[:, int(round(i * bw)):int(round((i + 1) * bw))]
        _, pos = per_row(crop, refine(crop, 6.1))
        s = scatter(pos)
        assert 0.0005 < s < 0.5, f"a blurred, noisy 6 px grating wanders {s:.3f} px"

        # The cross sheet, end to end: print both strips, blur the page HARDER
        # down than across, split it, read both, and check the ratio says so.
        # This is the whole claim of --cross -- that one sheet can tell the two
        # directions apart -- and it is the one thing a sheet of straight bars
        # cannot say about itself either.
        # Wider than 140 mm and at 300 dpi on purpose: a block is then wider
        # than the 10 mm gap, which is the case A4 makes and the case that
        # matters -- the down strip's blank block is a band of blank rows too,
        # and it is the LONGER one. An 80x120 sheet hides that, and hid it
        # until a real cross sheet came back off the glass (LAB 24.9).
        png = os.path.join(tmp, "cross.png")
        args = argparse.Namespace(paper="150x200", dpi=300, margin_mm=5.0,
                                  mm=15.0, out=png, cross=True)
        with contextlib.redirect_stdout(_io.StringIO()):
            do_print(args)
        page = trim_paper(load_ink(png)).astype(np.float64)
        soft = np.apply_along_axis(np.convolve, 0, page, np.ones(5) / 5, "same")
        across, down = split_cross(soft)
        assert abs(across.shape[0] - mm_px(15, 300)) < 8, (
            f"the across strip came back {across.shape[0]} rows tall, not "
            f"{mm_px(15, 300)}: the sheet was split somewhere else")
        assert down.shape[0] > 4 * across.shape[0], (
            f"the down strip is {down.shape[0]} rows of a {page.shape[0]}-row "
            f"page: the split landed inside it")
        with contextlib.redirect_stdout(_io.StringIO()):
            a, d = measure(across, "across"), measure(down.T, "down")
        for (spec, ca, _, _), (_, cd, _, _) in zip(a, d):
            if spec == "stroke" or ca < 0.1:
                continue
            assert cd < ca * 0.95, (
                f"{spec} px reads {cd:.3f} down against {ca:.3f} across, and "
                f"the page was blurred five pixels down and none across")
    print("grating selftest ok")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("print", help="a strip of gratings to print")
    pr.add_argument("-o", "--out", help="output PNG (default: strip.png)")
    pr.add_argument("--paper", default=PAPER_DEFAULT,
                    help=f"(default {PAPER_DEFAULT})")
    pr.add_argument("--dpi", type=int, default=DPI, help=f"(default {DPI})")
    pr.add_argument("--margin-mm", type=float, default=MARGIN_MM,
                    help=f"(default {MARGIN_MM:g})")
    pr.add_argument("--mm", type=float, default=50.0,
                    help="height of the strip in mm (default 50)")
    pr.add_argument("--cross", action="store_true",
                    help="print the strip twice on one sheet, the second one "
                         "turned, to measure the blur down the page against "
                         "the blur across it")

    rd = sub.add_parser("read", help="a scan of the strip -> contrast and noise")
    rd.add_argument("scan")
    rd.add_argument("--cross", action="store_true",
                    help="the scan is a --cross sheet: read both strips and "
                         "divide one by the other")

    sub.add_parser("selftest", help="draw the strip, blur it, read it, check it")

    args = p.parse_args()
    if args.cmd == "selftest":
        selftest()
    else:
        (do_print if args.cmd == "print" else do_read)(args)


if __name__ == "__main__":
    main()
