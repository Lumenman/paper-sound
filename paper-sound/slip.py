#!/usr/bin/env python3
"""Print a sheet whose lanes are gratings driven by real audio, and read it.

LAB 24.6 measured that a grating beats a stroke by 8 dB on this paper. It
measured it on bars that do not move, so it said nothing about the one thing a
moving grating can do that a stroke cannot: lose a cycle. A stroke that is
misread costs one sample. A grating whose phase is unwrapped down the lane and
slips costs every sample after it, by a whole period, to the end of the second.

The arithmetic answers that before any paper does, and the answer is that the
plain design does not work:

    slip.py slew ../out1.wav

    out1.wav at 15.85 px of excursion: the sample moves 0.34 px a row at the
    median, 5.16 px at the 99th, 25.05 px at the worst. A period of 4.9 px
    unwraps only while it moves less than 2.45 -- 6.6% of the rows are over.

paper_sound.py:333 says "the stroke moves ~0.2 px a row", and LAB 24.3 leaned
on that to call the unwrap trivial. 0.2 px is the median of a quiet file. The
tail is what an unwrap breaks on, and every wav in this project has one: 3.1%
of rows over the limit on the gentlest, 7.2% on the worst.

So the lane needs its cycle named rather than counted, and this sheet prints
both ways to be told apart on paper:

  plain     one grating of CYCLES[-1] periods per lane, phase unwrapped row by
            row. The control. It is expected to fail, and what is worth
            printing is HOW -- how many slips, how long the runs, whether a
            slip is audible or merely wrong.
  vernier   two lanes per second, gratings of 6 and 7 periods across the same
            width. Two cyclic phases whose difference completes one turn per
            lane width, so the position is absolute every row and nothing is
            unwrapped at all. LAB 24.3 named this and never priced it: it costs
            half the seconds on a sheet, which is the thing to weigh against
            the 8 dB.

    slip.py print ../out1.wav --kind vernier -o sheet_v.png
    slip.py read sheet_v_scan.png --kind vernier --truth ../out1.wav -o back.wav
    slip.py sim ../out1.wav --kind vernier      # no paper, both kinds priced
    slip.py selftest

`sim` is there because paper is the expensive way to learn arithmetic. Run it
first; print only what survives it. What it says today, on out1.wav at 6780 Hz
with the horizontal blur fixed at the sigma 1.08 px LAB 24.6 measured, the
noise at the 4.6/255 that same page implies, and the vertical blur swept:

    vertical sigma    plain    vernier    stroke    vernier - stroke
              0.00    -21.2      +41.8     +28.4          +13.3 dB
              0.50    -20.9      +30.6     +23.0           +7.7
              0.75    -22.3      +11.3     +14.5           -3.3
              1.08    -21.4       +2.9     +10.8           -7.9

`plain` is dead at every setting, as the slew table promised. `vernier` beats
the format it is trying to replace by 13 dB if the scanner's blur DOWN the page
is small, and loses to it by 8 dB if that blur is the same as the blur across
the page. The crossover is near 0.65 px, and the vernier's 13 dB is against one
lane a second where it spends two, so at equal paper it is nearer 10.

That one number has now been measured, and it says do not print this. LAB 24.9
put a cross sheet through two scanners: the blur DOWN the page is 1.09 px on
one and 1.27 on the other, against a crossover of 0.65. Both are the bottom row
of the table above -- the vernier at +2.9 against the stroke's +10.8. It loses,
and it loses before anything modulated has had the chance to slip a cycle.

Nor did the carriage kill it. The scanner whose glass is symmetric still reads
1.09, so most of that blur is the PRINT, isotropic, and no better flatbed buys
it back. A sharper printer might; nothing in this file can. So `sim` stays
useful and `print` stays unprinted, until LAB 24.9's next sheet says the blur
can be made smaller.

What this sheet does NOT settle: it prints no clocks, so it is scored against
the source with a lag search and not retimed. The carriage ripple that
pilot_timebase() takes out of an ordinary sheet is left in every number here,
which costs both kinds the same and flatters neither.
"""
import argparse

import numpy as np

from paper_sound import (BASELINE, DPI, HEADROOM, MARGIN_MM, PAPER_DEFAULT,
                         PITCH, highpass, lay_out as stroke_lay_out, mm_px,
                         paper_mm, read_curves_page, read_wav, render_page,
                         sheet_px, to_rate, trim_paper)
from grating import refine
from read_tracks import (find_tracks, ink_lean, inked_span, load_ink, resample,
                         write_wav)

# px of blank kept at each edge of a lane. Two lanes of grating with nothing
# between them are one grating, and find_tracks() reads the page's periodicity
# to cut it -- a page with no gutters has its period at the BAR, not the lane,
# and every lane on it is lost at once. Three pixels is two blurs wide.
GUARD = 3.0

# Periods across a lane's inked width, coarse first. They must be adjacent
# integers: the beat of W/n and W/(n+1) is exactly W, the whole lane, so the
# pair is absolute over the entire excursion and no wider pair would be. 6 and
# 7 put the fine grating at 4.67 px, which LAB 24.6 measured at 0.0500 px of
# noise against the stroke's 0.1263 -- the bottom of the V, near enough.
CYCLES = (6, 7)

# px of excursion, against the (PITCH - STROKE)/2 - MARGIN = 15.85 the stroke
# format gets. Deliberately short of the 16.35 the lane could hold: the vernier
# is absolute over exactly one lane width, so audio printed hard against that
# edge has its coarse estimate deciding between two cycles with no room for
# error, and a coarse estimate is what carries the noise of BOTH gratings. 14
# leaves 4.7 px of slack at the extremes and costs 1.1 dB. It is the knob to
# turn first if the paper says the vernier is picking wrong cycles.
AMP = 14.0

# The scan model `sim` uses. A GAUSSIAN of this sigma in px, not the box the
# rest of the project simulates with: a box of N has exact nulls at period N/k,
# and LAB 24.4 already caught one swallowing lambda 4.9 whole. This sheet's
# grating sits at 4.67 px, a box of 5 nulls it to nothing, and the read comes
# back at r = 0.35 for a reason that is entirely the model. Sigma 1.08 px is
# what LAB 24.6 fitted to nine measured contrast points on real paper.
#
# Applied along BOTH axes, and the vertical one turns out to BE the question.
# It mixes a row with the rows above and below, and at 4 px a row of slew those
# rows are a period away -- a stroke smeared with its neighbours still has a
# centroid, a grating smeared with its neighbours has its fundamental
# cancelled. Swept, this one number decides the whole idea: the vernier reads
# +41.8 dB against the stroke's +27.1 at vertical sigma 0, and +3.4 against
# +10.8 at 1.08. The crossover is near 0.6.
#
# And 1.08 px is the HORIZONTAL blur, measured off a strip of vertical bars,
# which is the only direction such a strip can measure. LAB 24.9 measured the
# vertical one on a cross sheet: 1.09 px on a scanner whose glass is symmetric
# and 1.27 on one whose carriage adds 0.47 px of its own. Hence --vblur, and
# hence a default that is if anything optimistic: the real page is 1.09 both
# ways at best.
SIM_SIGMA = 1.08
# Of 255, at full ink. LAB 24.6 read the paper at 9.3 +-2.3 of 255 with the ink
# at 128, so the noise against full ink is 2.3 * 255/128 = 4.6. The rest of the
# project simulates with 10, which is the older calibration and twice as harsh
# as the paper it was calibrated against.
SIM_NOISE = 4.6 / 255


def lane_ink():
    return PITCH - 2 * GUARD


def bars(pos, lo, w, n):
    """(left, right) column pairs for one lane's cyclic grating.

    `pos` is per row, in px of shift. The grating has exactly `n` periods
    across `w`, so shifting it by w is shifting it by nothing: the pattern is
    seamless at the wrap and the phase is the position, modulo one period,
    with no edge for the audio to run into.

    Each bar is drawn three times, at the wrap and one lane width either side,
    and clipped to the lane. Two of the three are empty for any given row and
    cost nothing but a subtraction; the third is the half of a bar that has
    just crossed the seam and has to come back on the other side.
    """
    lam = w / n
    out = []
    for k in range(n):
        c = lo + (pos + k * lam) % w
        for d in (0.0, w, -w):
            out.append((np.clip(c + d - lam / 4, lo, lo + w),
                        np.clip(c + d + lam / 4, lo, lo + w)))
    return out


def lay_out(signal, rows, nlanes, cycles):
    """Audio -> one page of grating lanes, rendered lane by lane.

    Lane by lane rather than through one render_page() call because a grating
    lane is 21 edge pairs where a stroke lane is one, and a page of them is a
    quarter of a gigabyte of edges held at once. A lane is 39 columns wide, so
    rendered alone it is nothing.
    """
    w = lane_ink()
    width = int(np.ceil(nlanes * PITCH)) + 1
    page = np.zeros((rows, width), np.float32)
    for k in range(nlanes):
        # The lane's SECOND, not the lane's number. `vernier` spends two lanes
        # on one second and they have to be the same second -- a pair carrying
        # two different seconds solves cleanly for a position that never
        # existed, and the read comes back at r = 0.02 with nothing in the
        # arithmetic to blame.
        j = k // len(cycles)
        s = signal[j * rows:(j + 1) * rows]
        if len(s) < rows:
            s = np.pad(s, (0, rows - len(s)))
        n = cycles[k % len(cycles)]
        x0 = k * PITCH
        lo = GUARD
        # Rendered in the lane's own columns and pasted, so render_page never
        # sees the page's width. The +w keeps `pos` positive before the modulo.
        # Centred in the lane, not at its edge: `vernier` hands back a
        # position in [0, w) and the audio must not straddle that seam, or the
        # trace steps by a whole lane in the middle of a quiet passage. AMP is
        # chosen so w/2 +- AMP clears both ends.
        cut = render_page(bars(AMP * s + w / 2, lo, w, n), 0.0,
                          int(np.ceil(PITCH)) + 1)
        c0 = int(x0)
        page[:, c0:c0 + cut.shape[1]] = np.maximum(
            page[:, c0:c0 + cut.shape[1]], cut[:, :width - c0])
    return page


def period(crop, n):
    """The period this lane actually carries, near the `n` it was drawn with.

    Not crop_width/n. The crop is cut on the lane's own ink and the ink has
    been through a printer and a platen, so its width is off by a few percent
    and the drawn period is off with it -- and a period wrong by 3% reads a
    ramp across the excursion 1.1 px wrong at the ends, which is eight times
    the noise the whole idea is competing on. Measured instead, by the same
    sweep grating.py puts a scale on the strip with: a lane prints its own
    ruler exactly as the strip did.

    Searched WIDE first, and that width is the whole trick. The crop is cut on
    ink, print and scan spill ink into the gutters, and the crop comes back
    10% wider than the lane was drawn -- so crop/n, the obvious first guess, is
    10% high and a search of +-8% around it does not contain the answer at all.
    It then returns the edge of its own window, the trace is scaled 10% wrong,
    and the vernier picks cycles off a coarse reading that drifts a whole
    period across the excursion. Measured: 5.126 px found where 4.671 was
    printed, and a page that should read +10 dB reading -4.
    """
    lam = refine(crop, crop.shape[1] / n, span=0.25, steps=201)
    return refine(crop, lam, span=0.02, steps=81)


def phase(crop, lam):
    """Fundamental phase of a cyclic grating, per row, in px of shift.

    Wrapped into one period by construction -- this is the number the plain
    kind has to unwrap and the vernier kind does not.
    """
    x = np.arange(crop.shape[1])
    win = np.hanning(crop.shape[1] + 2)[1:-1]
    k = 2 * np.pi / lam
    a = crop @ (np.cos(k * x) * win)
    b = crop @ (np.sin(k * x) * win)
    return np.arctan2(b, a) * lam / (2 * np.pi) % lam


def solve(p_coarse, p_fine, lam_c, lam_f, w):
    """Two cyclic phases -> absolute position in [0, w).

    The two gratings advance at different rates, so the DIFFERENCE of their
    phases, in turns, completes exactly one turn as the pattern moves one lane
    width. That difference is the coarse reading: it is absolute, and it is
    noisy, carrying the noise of both gratings multiplied by the ratio. The
    fine grating then says where inside its own period the pattern sits, and
    the coarse reading only has to be right to half a fine period to pick which
    period that is. LAB 24.6 measured the fine phase at 0.05 px, which puts the
    coarse at about 0.35 and the decision at 6 sigma.
    """
    turns = (p_fine / lam_f - p_coarse / lam_c) % 1.0
    coarse = turns * w

    # Which period of the fine grating the coarse reading lands in. The
    # decision is a rounding, so it is only safe while what is being rounded
    # sits near a whole number -- and nothing here makes it. The two lanes are
    # cut by their own ink, so each phase carries its own constant offset, and
    # their difference lands the rounding wherever it lands. Half the time
    # that is near .5, where the coarse reading's own noise flips the answer
    # by a whole period on every other row: measured, a pair that reads 0.04
    # px of phase noise comes back with 4.55 px of position error, which is
    # exactly one period of flapping.
    #
    # The offset is constant down the lane, so the lane measures it itself: a
    # circular mean of the fractional part, which is the right mean for a
    # quantity that wraps. Subtracted before the rounding, it puts the decision
    # in the middle of its own bin and leaves the noise the 6 sigma it should
    # have had. This is the calibration a real sheet needs and a clean one
    # never shows -- on a page drawn and not printed the offset is zero.
    u = (coarse - p_fine) / lam_f
    off = np.angle(np.mean(np.exp(2j * np.pi * u))) / (2 * np.pi)
    pos = ((np.round(u - off) + off) * lam_f + p_fine) % w

    # And the seam. `pos` is absolute within one lane width and the lane width
    # is a circle: the two phases carry their own constant offsets, so where
    # zero falls in the audio's range is an accident of where the crops were
    # cut. Land it inside the excursion and every row on the far side of it
    # reads a whole lane width away -- three of four lane pairs did, at 5.9 px
    # rms with only 3% of rows past a period, which is the shape of a wrap and
    # not of noise. Turned so the lane's own median sits at zero, the seam goes
    # to the far side of the circle, where AMP has already left room -- and
    # the room IS the answer. AMP is short of w/2 on purpose, so a lane's audio
    # can never fill the circle, and the arc it never visits is where the seam
    # belongs. Found as the longest run of empty bins rather than by a mean or
    # a median: both of those are points on a line, and on a circle they land
    # in the wrong place exactly when the audio is loud, which is when it
    # matters. Bins rather than raw gaps because a handful of rows that picked
    # the wrong cycle land anywhere, and one of them inside the dead zone would
    # otherwise hand the seam to the audio.
    return (pos - seam(pos, w)) % w - w / 2


def seam(pos, w, bins=64):
    """Where to cut the circle: the middle of the widest arc the audio misses."""
    h, _ = np.histogram(pos % w, bins=bins, range=(0, w))
    empty = h <= max(1, len(pos) // (200 * bins))
    if not empty.any():
        # No dead zone at all: the audio has been round the whole lane and the
        # position is ambiguous by a lane width for the rest of this second.
        # AMP is the knob; nothing here can recover it.
        return 0.0
    best = run = start = 0
    at = 0
    for i in range(2 * bins):            # twice round, so a run may wrap
        if empty[i % bins]:
            run = run + 1
            if run > best and i < bins + start:
                best, at = run, i - run + 1
        else:
            run, start = 0, i + 1
    return ((at + best / 2) % bins) * w / bins


def lane_crops(ink, nlanes=None):
    """The scan cut into lane-wide crops, plus the whole-row shift taken out.

    find_tracks() gives straight column slices, and a sheet on a platen is not
    straight: 17.7 px of shear across the page was measured on the crooked
    sheet, which is half a lane. A crop that straight would hold two lanes at
    the bottom and neither in the middle. The shear is already measured by
    ink_lean() -- geometry off the ink block, asking nothing about content --
    so every row is rolled back by it before anything is read.

    The roll is a whole number of pixels and it is handed back with the crops,
    because a phase is a position and a position that has been moved by a
    rounded pixel is wrong by up to half of one. Measured on a page with no
    skew at all, where the lean estimate lands near zero and rolls a scattering
    of rows by exactly 1: 12% of rows came back 1.02 px out, in a spike rather
    than a tail, while the other 88% sat at 0.015. A stroke reader never sees
    this because lane_centroid() rides a sub-pixel ramp instead of rolling; a
    grating reader has to put the pixel back.
    """
    lean = min(ink_lean(ink), key=abs) if len(ink_lean(ink)) else 0.0
    rows = np.arange(ink.shape[0])
    shift = np.round(lean * (rows - len(rows) / 2)).astype(int)
    cols = (np.arange(ink.shape[1]) + shift[:, None]) % ink.shape[1]
    straight = ink[rows[:, None], cols]
    lanes = find_tracks(straight)
    if nlanes and len(lanes) != nlanes:
        print(f"  NOTE: {len(lanes)} lanes cut where the sheet was printed "
              f"with {nlanes} -- the numbers below are against what was found")
    # Cut on each lane's OWN ink rather than on a share of the pitch. The
    # gutters are what find_tracks() cut between, and where inside them it put
    # the cut is its business; what a phase needs is the inked width, and a
    # lane averaged down the page is flat across its grating and paper in its
    # gutters, so the ink says where it starts.
    out = []
    for x0, x1 in lanes:
        a, b = inked_span(straight[:, x0:x1].mean(0))
        out.append(straight[:, x0 + a:x0 + b].astype(np.float64))
    return out, shift.astype(np.float64)


def read_lanes(crops, shift, kind):
    """Lane crops -> one position trace per SECOND, in px.

    `shift` is what lane_crops() rolled each row by, added straight back into
    the phase: the roll moved the ink, the phase measures where the ink is, so
    the two cancel exactly and nothing is left rounded.
    """
    if kind == "plain":
        out = []
        for c in crops:
            lam = period(c, CYCLES[-1])
            p = (phase(c, lam) + shift) % lam
            out.append(np.unwrap(p * 2 * np.pi / lam) * lam / (2 * np.pi))
        return out
    out = []
    for a, b in zip(crops[::2], crops[1::2]):
        lam_c, lam_f = period(a, CYCLES[0]), period(b, CYCLES[1])
        out.append(solve((phase(a, lam_c) + shift) % lam_c,
                         (phase(b, lam_f) + shift) % lam_f,
                         lam_c, lam_f, lam_f * CYCLES[-1]))
    return out


def slips(trace, truth, lam):
    """(rows adrift, longest run) of a trace against the audio it was printed from.

    A slip is not a bad sample. It is a step of one period that stays, so what
    is counted is rows sitting more than half a period from where the source
    put them, once each is centred on its own median -- and the run length is
    the number that matters, because a run IS the damage.
    """
    a = trace - np.median(trace)
    b = truth[:len(a)] * AMP
    b = b - np.median(b)
    bad = np.abs(a - b) > lam / 2
    run = best = 0
    for v in bad:
        run = run + 1 if v else 0
        best = max(best, run)
    return int(bad.sum()), best


def db(a, b, lag=8):
    """The project's own metric between a read and its source."""
    n = min(len(a), len(b)) - 2 * lag
    a, b = highpass(np.asarray(a, float), BASELINE), highpass(np.asarray(b, float), BASELINE)
    rs = [np.corrcoef(a[lag + d:lag + d + n], b[lag:lag + n])[0, 1]
          for d in range(-lag, lag + 1)]
    r = float(np.clip(max(rs), -0.999999, 0.999999))
    return 10 * np.log10(r * r / (1 - r * r)), r


def prepared(path, rate, seconds=None):
    """The wav, at the sheet's rate, scaled the way paper_sound.py prints it."""
    signal, sr = read_wav(path)
    signal = to_rate(signal, sr, rate)
    scale = np.percentile(np.abs(signal), 99.9) * HEADROOM
    signal = np.clip(signal / scale, -1.0, 1.0) if scale else signal * 0.0
    return signal[:seconds * rate] if seconds else signal


def report(crops, kind, truth, rate, label=""):
    """Read one sheet's crops both ways and say what the lanes did."""
    crops, shift = crops
    traces = read_lanes(crops, shift, kind)
    step = 1 if kind == "plain" else 2
    lam = float(np.mean([period(c, CYCLES[-1]) for c in crops[step - 1::step]]))
    song = np.concatenate([resample(t - np.median(t), rate) for t in traces])
    d, r = db(song, truth[:len(song)])
    adrift, longest, secs = 0, 0, len(traces)
    for i, t in enumerate(traces):
        n, run = slips(t, truth[i * rate:(i + 1) * rate], lam)
        adrift += n
        longest = max(longest, run)
    print(f"  {label + kind:>10}  {secs:3d} s  {d:6.2f} dB (r {r:.4f})  "
          f"fine {lam:.2f} px  adrift {adrift / (secs * rate) * 100:6.2f}% of rows"
          f"  longest run {longest:5d}")
    return song, d


def gaussian(sigma):
    x = np.arange(-max(1, int(np.ceil(3 * sigma))), max(1, int(np.ceil(3 * sigma))) + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def scan_model(page, rng, sigma=SIM_SIGMA, vsigma=None, noise=SIM_NOISE):
    """Print and scan, roughly: a gaussian blur each way, then noise."""
    ink = page.astype(np.float64)
    if sigma:
        ink = np.apply_along_axis(np.convolve, 1, ink, gaussian(sigma), "same")
    if vsigma is None:
        vsigma = sigma
    if vsigma:
        ink = np.apply_along_axis(np.convolve, 0, ink, gaussian(vsigma), "same")
    return np.clip(ink + rng.normal(0, noise, ink.shape), 0, 1)


def do_slew(args):
    signal, sr = read_wav(args.audio)
    d = np.abs(np.diff(signal)) * AMP
    w = lane_ink()
    print(f"{args.audio}: {sr} Hz, {len(signal) / sr:.0f} s, at {AMP:g} px of "
          f"excursion")
    for q in (50, 90, 99, 99.9, 100):
        print(f"  moves {np.percentile(d, q):6.2f} px a row at the "
              f"{q:g}th percentile")
    print(f"  {'periods':>9} {'lam px':>7} {'lam/2':>7} {'rows over':>10}")
    for n in range(4, 10):
        lam = w / n
        print(f"  {n:9d} {lam:7.2f} {lam / 2:7.2f} "
              f"{(d > lam / 2).mean() * 100:9.2f}%")
    print("  a row over the limit is a row an unwrap may slip on. `vernier` "
          "does not unwrap and does not care about this table")


def do_print(args):
    from PIL import Image

    width_px, rate = sheet_px(args.paper, args.dpi, args.margin_mm)
    nlanes = int(width_px // PITCH)
    per = 1 if args.kind == "plain" else len(CYCLES)
    signal = prepared(args.audio, rate, nlanes // per)
    cycles = CYCLES[-1:] if args.kind == "plain" else CYCLES
    page = lay_out(signal, rate, nlanes, cycles)

    paper_w, paper_h = paper_mm(args.paper)
    sheet_w, sheet_h = mm_px(paper_w, args.dpi), mm_px(paper_h, args.dpi)
    out_page = np.zeros((sheet_h, sheet_w), np.float32)
    y, x = (sheet_h - rate) // 2, (sheet_w - page.shape[1]) // 2
    out_page[y:y + rate, x:x + page.shape[1]] = page
    out = args.out or f"sheet_{args.kind}.png"
    Image.fromarray(255 - np.round(out_page * 255).astype(np.uint8)).save(
        out, dpi=(args.dpi, args.dpi))
    print(f"{out}: {nlanes} lanes of {PITCH:g} px, {nlanes // per} seconds at "
          f"{rate} Hz, {args.kind}")
    print(f"  print at 100%, scan at {args.dpi} dpi greyscale with every "
          f"adjustment off -- this measures edges")


def do_read(args):
    ink = trim_paper(load_ink(args.scan)).astype(np.float64)
    ink = ink / max(ink.max(), 1.0)
    crops = lane_crops(ink)
    rate = ink.shape[0]
    print(f"{args.scan}: {ink.shape[1]}x{rate} px of ink, {len(crops[0])} lanes")
    truth = prepared(args.truth, rate)
    song, _ = report(crops, args.kind, truth, rate)
    if args.out:
        write_wav(args.out, song, rate)
        print(f"  {args.out}")


def control(signal, rate, nlanes, rng, args):
    """The same audio as an ordinary sheet, through the same model.

    Without this the dB above have no scale. LAB 24.6 is the whole reason it is
    here: a number measured against a reference in another pass is a number
    divided by a different scanner. This one is drawn by paper_sound's own
    lay_out() and read by its own read_curves_page(), so what is being compared
    is the format, not two readers.
    """
    edges, _ = stroke_lay_out(signal[:nlanes * rate], rate, nlanes, PITCH, 0.0)
    page = render_page(edges, PITCH, int(np.ceil((nlanes + 2) * PITCH)))
    ink = scan_model(page, rng, args.blur, args.vblur, args.noise / 255)
    song, sr, n = read_curves_page(ink * 255)
    d, r = db(song, signal[:len(song)])
    print(f"  {'stroke':>10}  {n:3d} s  {d:6.2f} dB (r {r:.4f})  "
          f"{'the format as it stands, same audio, same model':>52}")


def do_sim(args):
    rng = np.random.default_rng(7)
    width_px, rate = sheet_px(args.paper, args.dpi, args.margin_mm)
    rate = rate // args.thin
    nlanes = args.lanes
    print(f"{args.audio} through the scan model at {rate} Hz, {nlanes} lanes, "
          f"gaussian blur sigma {args.blur:g} across and "
          f"{args.blur if args.vblur is None else args.vblur:g} down, noise "
          f"{args.noise:g}/255")
    for kind in (("plain", "vernier") if args.kind == "both" else (args.kind,)):
        per = 1 if kind == "plain" else len(CYCLES)
        signal = prepared(args.audio, rate, nlanes // per)
        cycles = CYCLES[-1:] if kind == "plain" else CYCLES
        page = lay_out(signal, rate, nlanes, cycles)
        crops = lane_crops(scan_model(page, rng, args.blur, args.vblur,
                                      args.noise / 255), nlanes)
        report(crops, kind, prepared(args.audio, rate), rate)
    control(prepared(args.audio, rate, nlanes), rate, nlanes, rng, args)


def selftest():
    rng = np.random.default_rng(7)
    w = lane_ink()
    rows, nlanes = 600, 4

    # A ramp across the whole excursion, drawn and read with nothing in
    # between: both kinds must give back the ramp. This is the arithmetic of
    # bars(), phase() and solve() and nothing else.
    ramp = np.linspace(-1, 1, rows)
    for kind, cycles in (("plain", CYCLES[-1:]), ("vernier", CYCLES)):
        page = lay_out(np.tile(ramp, nlanes), rows, nlanes, cycles)
        crops, shift = lane_crops(page.astype(np.float64), nlanes)
        got = read_lanes(crops, shift, kind)[0]
        got = got - got.mean()
        want = AMP * ramp - (AMP * ramp).mean()
        err = float(np.abs(got - want)[20:-20].max())
        assert err < 0.15, f"{kind} reads a ramp {err:.3f} px wrong"

    # The claim the sheet exists to test, in one line each way. A signal that
    # jumps more than half a period between rows must break the unwrap and
    # must NOT break the vernier -- that is the whole difference between the
    # two kinds, and if it ever stops holding here the sheet is measuring
    # nothing.
    # A chirp, not a square wave. A signal that only ever sits at two extremes
    # is genuinely ambiguous on a cyclic lane -- nothing in the ink says
    # whether it jumped the short way round or the long way -- so testing
    # against one tests a question with no answer. A chirp visits everything on
    # its way and ends up moving further than half a period per row, which is
    # the condition that matters.
    t = np.arange(rows) / rows
    fast = 0.9 * np.sin(2 * np.pi * rows * 0.12 * t ** 2)
    for kind, cycles, want_slips in (("plain", CYCLES[-1:], True),
                                     ("vernier", CYCLES, False)):
        page = lay_out(np.tile(fast, nlanes), rows, nlanes, cycles)
        crops, shift = lane_crops(page.astype(np.float64), nlanes)
        trace = read_lanes(crops, shift, kind)[0]
        lam = period(crops[-1], CYCLES[-1])
        adrift, _ = slips(trace, fast, lam)
        if want_slips:
            assert adrift > rows // 10, (
                f"the unwrap survived {rows} rows of full-scale jumps -- "
                f"only {adrift} adrift, so this test is not testing it")
        else:
            assert adrift < rows // 50, (
                f"the vernier lost {adrift} of {rows} rows on a signal it is "
                f"supposed to be absolute against")
    print("slip selftest ok")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def paper_args(q):
        q.add_argument("--paper", default=PAPER_DEFAULT)
        q.add_argument("--dpi", type=int, default=DPI)
        q.add_argument("--margin-mm", type=float, default=MARGIN_MM)
        return q

    sl = sub.add_parser("slew", help="how fast this audio moves, against the "
                                     "period an unwrap could survive")
    sl.add_argument("audio")

    pr = paper_args(sub.add_parser("print", help="a sheet of grating lanes"))
    pr.add_argument("audio")
    pr.add_argument("-o", "--out")
    pr.add_argument("--kind", choices=("plain", "vernier"), default="vernier")

    rd = sub.add_parser("read", help="a scan of that sheet -> dB and slips")
    rd.add_argument("scan")
    rd.add_argument("--truth", required=True, help="the wav it was printed from")
    rd.add_argument("--kind", choices=("plain", "vernier"), default="vernier")
    rd.add_argument("-o", "--out", help="write the read back as a wav")

    sm = paper_args(sub.add_parser("sim", help="both kinds through the scan "
                                               "model, no paper"))
    sm.add_argument("audio")
    sm.add_argument("--kind", choices=("plain", "vernier", "both"),
                    default="both")
    sm.add_argument("--lanes", type=int, default=8)
    sm.add_argument("--blur", type=float, default=SIM_SIGMA,
                    help=f"gaussian sigma in px (default {SIM_SIGMA:g}, "
                         f"measured -- LAB 24.6)")
    sm.add_argument("--vblur", type=float, default=None,
                    help="gaussian sigma DOWN the page (default: same as "
                         "--blur). Never measured on paper; this is the "
                         "number the whole idea turns on")
    sm.add_argument("--noise", type=float, default=SIM_NOISE * 255,
                    help=f"of 255 (default {SIM_NOISE * 255:g})")
    sm.add_argument("--thin", type=int, default=4,
                    help="rows per lane divided by this, to keep sim quick "
                         "(default 4)")

    sub.add_parser("selftest")

    args = p.parse_args()
    {"slew": do_slew, "print": do_print, "read": do_read, "sim": do_sim,
     "selftest": lambda a: selftest()}[args.cmd](args)


if __name__ == "__main__":
    main()
