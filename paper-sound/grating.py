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
    y, x = (sheet_h - rows) // 2, (sheet_w - width) // 2
    page[y:y + rows, x:x + width] = strip
    out = args.out or "strip.png"
    Image.fromarray(255 - np.round(page * 255).astype(np.uint8)).save(
        out, dpi=(args.dpi, args.dpi))
    print(f"{out}: {len(BLOCKS)} blocks of {bw:.0f} px, ink {width}x{rows} px "
          f"= {width * 25.4 / args.dpi:.0f}x{args.mm:g} mm at {args.dpi} dpi")
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


def do_read(args):
    ink = trim_paper(load_ink(args.scan)).astype(np.float64)
    rows, width = ink.shape
    bw = width / len(BLOCKS)
    cuts = [ink[:, int(round(i * bw)):int(round((i + 1) * bw))]
            for i in range(len(BLOCKS))]

    solid = float(np.mean([middle(cuts[i]).mean()
                           for i, s in enumerate(BLOCKS) if s == "solid"]))
    blank = middle(cuts[BLOCKS.index("blank")])
    if solid <= 0:
        raise SystemExit(f"{args.scan}: no ink -- is this the strip?")

    # The strip prints its own ruler: the reference block's period IS the
    # format's PITCH, so what it measures as is the scan's scale. Nothing here
    # has to be told the dpi it was printed at, or the one it came back at.
    ref = cuts[BLOCKS.index("stroke")]
    pitch_px = refine(ref, PITCH, span=0.06, steps=241)
    scale = pitch_px / PITCH
    narrow = APERTURE * scale

    print(f"{args.scan}: {width}x{rows} px of ink, {len(BLOCKS)} blocks of "
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
        return

    fine = [r for r in out if r[0] != "stroke" and r[3] > 0]
    if not fine:
        return
    spec, contrast, lam, sigma = min(fine, key=lambda r: r[3])
    swing = 2 * ((PITCH - STROKE) / 2 - 1.5)
    print(f"  best: {spec:g} px, {20 * np.log10(ref_sigma / sigma):+.2f} dB "
          f"against the stroke this same sheet prints")
    print(f"  a lane holds {PITCH / spec:.1f} cycles of it, and the phase wraps "
          f"{swing / spec:.1f} times across the format's {swing:.1f} px of "
          f"excursion -- so the sample has to move less than {spec / 2:.2f} px "
          f"a row for the unwrap to hold. It moves about 0.2")


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

    rd = sub.add_parser("read", help="a scan of the strip -> contrast and noise")
    rd.add_argument("scan")

    sub.add_parser("selftest", help="draw the strip, blur it, read it, check it")

    args = p.parse_args()
    if args.cmd == "selftest":
        selftest()
    else:
        (do_print if args.cmd == "print" else do_read)(args)


if __name__ == "__main__":
    main()
