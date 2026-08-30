#!/usr/bin/env python3
"""Print audio onto paper as curves, and read a scan of that paper back.

The format is its own documentation: **one lane is one second, one row is one
sample**, so the sheet's printable height in pixels IS the sample rate and its
width is how many seconds fit. Nothing is encoded anywhere -- pitch, rate, lane
count and skew are all measured off the sheet when it is read back. There is no
header, no checksum and no metadata; a sheet is a picture of its own contents.

`print --rows 2` draws each sample across two rows instead of one. The ink's
slope per row halves, a lane holds half a second, and a sheet holds half the
audio -- paper spent on quality, worth +1.9 dB measured. That is not encoded
either: the clocks are printed with a period in SAMPLES, so on paper their
period is that many rows TIMES the rows per sample, and the reader divides what
it measures. `read` takes no flag, and a sheet printed before this existed
measures as 1 and reads exactly as it always did.

A lane carries its second as a stroke of fixed width wandering inside it: the
sample is the stroke's POSITION, read as the ink's centroid across the lane.
Position rather than area, because a centroid is measured against the lane's
own centre and is blind to how much ink landed, so print density and exposure
drop out of the answer. That blindness holds only while the scanner's response
is linear -- see retouched().

A sheet is bracketed by two clock lanes: a plain tone, printed like any other
second and read back like any other second, one at each end of the field.
They carry the one number the sheet cannot otherwise give up. A scanner does
not advance one row per row, and the ripple in its carriage is a timing error
shared by every lane -- invisible to every other measurement a sheet can carry,
because it is purely vertical and leaves no horizontal trace. See
pilot_timebase(). The two are read as the ends of a line rather than averaged,
because they disagree about the slope and agree about the wiggle -- see
retime(). They are also printed at different rates, which breaks the one
symmetry a sheet has: turned 180 degrees it is otherwise identical, so which
clock reads back first says which way up it went on the glass, and an
upside-down scan turns itself -- see pilot_retime().

Two corrections, then: the clocks, and the skew estimate in read_curves_page().
Do not read a single number for either as a property of this reader. Measured
the same way on two printed and scanned sheets:

                      skew estimate   clock retiming   how the sheet lay
    the crooked sheet    +2.34 dB        +1.44 dB      17.7 px of skew
    the rippled sheet    +0.09 dB        +4.88 dB      4.9 px, 9 px of ripple

Opposite answers, for a mechanical reason: each correction is worth exactly as
much of its own defect as that sheet happened to carry, and neither defect is
knowable until the sheet is on the platen. They are insurance at a fixed
premium -- nothing for the skew, two lanes for the clocks -- against a payout
that has run from 0.1 dB to 5 dB so far. The case for keeping both is the worst
sheet, not the average one. Those two sheets are named throughout this file.

    paper_sound.py print song.wav -o sheet
    paper_sound.py print song.wav --paper letter --margin-mm 12
    paper_sound.py read sheet.png -o back.wav
    paper_sound.py read sheet1.png sheet2.png -o back.wav
    paper_sound.py selftest

Printing: at 100%, actual size, no "fit to page". The PNG carries its dpi, so a
sane print dialog gets it right on its own. If a sheet comes out faint, print
it again darker rather than fixing it afterwards.
Scanning: greyscale at the same dpi with every adjustment off -- no
auto-contrast, no levels, no white or black point, no sharpening, no deskew,
not bitonal. A white point is the expensive one: it can cost 8.6 dB in silence
and cannot be undone. See retouched().
"""
import argparse
import os
import wave

import numpy as np

from read_tracks import (HEADROOM, cumsum_at, declick, find_tracks, highpass,
                         ink_lean, ink_rows, load_ink, odd_lane, resample,
                         write_wav)

# ------------------------------------------------------------------- defaults
# Every default lives here; the command line only overrides them. Change a
# number in this block and both `print` and `read` follow.

PAPER_DEFAULT = "a4"
DPI = 600         # print and scan resolution
MARGIN_MM = 5.0   # paper the printer cannot reach, per edge
PITCH = 38.7      # px, lane pitch: 120 lanes across A4 at 600 dpi
COMB = 32         # empty histogram bins that mean a scan has been retouched

STROKE = 4.0    # px, width of a curve's stroke
MARGIN = 1.5    # px of lane kept clear at the extremes of the excursion
STICKY = 11     # rows the window's aim leans on, dneedle's stickyFact; 0 = off
APERTURE = 2 * STROKE   # px of lane the centroid window looks at, centred on
                        # the trace. The one live tuning knob in the reader,
                        # and the only one whose best value is a property of
                        # the scan rather than of the format: on a sheet that
                        # never met a printer, 8 px scores +31.10 dB and 3 px
                        # scores +13.00; on a printed and scanned one, 3 px
                        # scores +1.95 and 8 px +1.80. Wide sees all of the
                        # stroke, narrow rejects
                        # the grime beside it. The default sits wide because
                        # the loss is not symmetric -- too narrow on a clean
                        # read costs 18 dB, too wide on a dirty one costs 0.15.

BASELINE = 331  # samples of moving average subtracted. Measured, that is
                # -3 dB at 8.8 Hz on an A4 sheet at 600 dpi, not the 20 Hz this
                # line said for a long time.
                #
                # Both of these are SAMPLES, and a sheet's sample rate is its
                # printable height in pixels, so both move with the paper and
                # the dpi. Across everything this format can be printed on --
                # A4 at 300, 600 and 1200 dpi, A5, A3 -- the corner lands
                # between 4.4 and 17.7 Hz and the join spans 1.2 to 4.9 ms.
                # That is all below hearing and all still a bump rather than a
                # click, which is why they are plain numbers and not
                # sr // 20 | 1 and sr // 400 (which would give 329 and 16 here
                # and are what to write if a sheet ever needs them to hold).
SKEW_GAP = 15   # px of skew across the page that the fit and the sheet's own
                # ink edges may disagree by before it is called out; lane_drift()
STRIPS = 8      # strips of rows the page's bend is fitted in, lane_drift()
DECLICK = 16    # samples a lane join is spread over: 2.4 ms at 600 dpi on A4
DESPECKLE = 0   # multiple of the median step a sample may jump; 0 = off
PILOT = 32      # samples per cycle of the first clock lane; 0 = no clocks
PILOT_END = 20  # the last clock's cycle -- deliberately not PILOT. A sheet is
                # otherwise symmetric under a 180 degree turn, so which rate
                # reads back first is the one mark of which way up it lay.
                # See pilot_retime().
PILOT_AMP = 0.5 # a clock's excursion, as a fraction of the lane's full swing
ROWS = (1, 2)   # rows per sample a sheet may carry -- see `print --rows`. The
                # clocks are what say which: a clock's period on paper is its
                # own rate times this, so the candidates are 20, 32, 40 and 64
                # rows. 3 does not fit the ladder (32x3 against 20x5 is a 4%
                # gap) and 4 leaves 30 seconds on a sheet of A4.
DEC_TAPS = 16   # half-length of the windowed sinc to_rate() and thin() share,
                # in CYCLES of its own cutoff -- lowpass() turns that into
                # samples, because a kernel is only as good as the number of
                # cycles it holds. 8 and 32 measure the same (24.1 dB against
                # 24.5 on a digital sheet), so the ceiling is the straight-line
                # stretch `print` draws with and not the filter's skirt.
PILOT_SLACK = 1.118  # how far off its printed period a clock may read, as a
                # ratio. The closest pair of candidates above is 1.25 apart, so
                # half of that in log is all the slack there can be before a
                # clock is ambiguous. Real scans come in 3.1% off and the bench
                # 0.1%, so the margin is a factor of four.

PAPER = {           # sheet sizes, mm -- one unit for everything physical here
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
    "a5": (148.0, 210.0),
    "letter": (215.9, 279.4),
    "legal": (215.9, 355.6),
}

def read_wav(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path}: no such file")
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
        return x / 32768.0, w.getframerate()


def render_page(edges, pitch, width, margin=8):
    """Draw anti-aliased ink from a list of per-track (left, right) column pairs.

    Each entry is a pair of arrays holding, for every row, the absolute column
    where that track's ink starts and ends. Partial pixels at both ends carry
    fractional coverage, which is the entire source of sub-pixel information
    downstream -- render it with a hard round here and the bench would be
    measuring its own quantisation.
    """
    rows = len(edges[0][0])
    page = np.zeros((rows, width), np.float32)
    for left, right in edges:
        lo = max(0, int(np.floor(left.min())) - margin)
        hi = min(width, int(np.ceil(right.max())) + margin)
        idx = np.arange(lo, hi)[None, :]
        cover = np.minimum(idx + 1, right[:, None]) - np.maximum(idx, left[:, None])
        page[:, lo:hi] += np.clip(cover, 0.0, 1.0)
    return np.clip(page, 0.0, 1.0)


def lay_out(signal, sr, ntracks, pitch, skew, stroke=STROKE, edge=None):
    """One second of audio per lane -> the ink edges of each track.

    `edge` is the blank kept outside the outermost lane. It defaults to a
    pitch, because ink hard against column zero leaves the outermost lane
    clipped and silently dropped, and a dropped lane shifts every later second
    by one, which scores as noise. Printing on paper the sheet's own margin is
    that blank already, so there `edge` is only what the margin falls short of.
    """
    amp = (pitch - stroke) / 2 - MARGIN
    y = np.arange(sr)
    drift = skew * y
    edges = []

    start = pitch if edge is None else edge
    for k in range(ntracks):
        s = signal[k * sr:(k + 1) * sr]
        pos = start + k * pitch + pitch / 2 + drift + amp * s
        edges.append((pos - stroke / 2, pos + stroke / 2))
    return edges, amp


def box1d(a, n, axis):
    if n < 2:
        return a
    a = np.moveaxis(a, axis, -1)
    pad = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(n // 2, n - 1 - n // 2)], mode="edge")
    c = np.cumsum(pad, axis=-1)
    c = np.pad(c, [(0, 0)] * (c.ndim - 1) + [(1, 0)])
    return np.moveaxis((c[..., n:] - c[..., :-n]) / n, -1, axis)


def lane_centroid(ink, x0, x1, span, drift, narrow=2 * STROKE, sticky=STICKY,
                  absolute=False):
    """Ink centroid of one lane, per row, relative to the lane's own centre.

    `absolute` hands back the trace in the page's own columns, as
    `(rows, column per row)`, instead of centring it on its own median. That
    is the frame cut_lanes.traces() aims its aperture in: the same crop, the
    same rewritten left edge, the same mid-page y. Rebuilding it on the other
    side of the call is how the mask and the wav disagree the day this frame
    moves.
    """
    # The leftmost lane's cut is clipped at the ink's own edge -- trim_paper
    # cuts the page to the ink, so the lane's true left boundary lies off it.
    # Its centre is anchored on the right cut, which survived; anchor it on
    # x0 + span/2 like every other lane and the window sits the whole
    # truncation wide of the stroke. The right edge needs no twin: a centre is
    # measured from x0 anyway. Found by the left clock: read at 20 samples a
    # cycle it loses the stroke at the tone's far swing, where the slower 32
    # forgave the same offset for years.
    if x0 == 0 and x1 - x0 < span:
        x0 = x1 - span
    # `drift` is either a slope in px per row, or the page's own bend already
    # sampled row by row -- what lane_drift() hands back once one line stops
    # describing the sheet. Both are the same thing to everything below: an
    # offset per row, zero at the lane's middle.
    bent = np.ndim(drift) > 0
    # The crop has to hold the lane wherever the skew takes it: the window
    # rides `drift` away from the lane's mid-page position, and a sheet that
    # drifts three pitches down its height carries the window a lane and a half
    # outside a crop cut to the lane's own columns.
    reach = int(np.ceil((np.ptp(drift) if bent else abs(drift) * len(ink)) / 2))
    off = max(x0 - span - reach, 0)
    lane = ink[:, off:x1 + span + reach].astype(np.float64)
    rows = ink_rows(lane)      # this lane's own ink, not the page's rows
    if rows is None or rows[1] - rows[0] < 2:
        return None
    lane = lane[rows[0]:rows[1]]

    # Moments in the crop's own columns, not the page's. Mixing the two leaves
    # every centroid offset by the crop's origin -- which a median subtraction
    # hides for as long as the estimate is only ever the answer, and which puts
    # the window a whole lane wide of the stroke the moment the estimate is
    # used to aim one.
    x = np.arange(lane.shape[1], dtype=np.float64)
    cs = np.pad(np.cumsum(lane, axis=1), ((0, 0), (1, 0)))
    cx = np.pad(np.cumsum(lane * x, axis=1), ((0, 0), (1, 0)))

    # Centred on mid-page: x0 comes from the column profile averaged down the
    # whole sheet, so it is where the lane sits halfway down, not at its top.
    # Ramp from row zero instead and the window sits half the total drift wide
    # of the lane at both ends of the page.
    y = np.arange(len(lane)) - len(lane) / 2
    if bent:
        # Sampled from the lane's own first inked row, and centred like the
        # ramp it replaces: a lane's ink starts within a row or two of the
        # page's, and the median subtraction below takes the rest.
        ramp = np.asarray(drift, float)
        if len(ramp) < len(lane):
            ramp = np.pad(ramp, (0, len(lane) - len(ramp)), mode="edge")
        ramp = ramp[:len(lane)]
        ramp = ramp - ramp.mean()
    else:
        ramp = drift * y
    lo = (x0 - off) + ramp
    den = cumsum_at(cs, lo + span) - cumsum_at(cs, lo)
    num = cumsum_at(cx, lo + span) - cumsum_at(cx, lo)
    # Relative to the window's own centre, not to the page. The window rides
    # the skew, so measuring against it subtracts the skew exactly; take the
    # centroid in absolute columns instead and the whole ramp -- on a badly
    # skewed sheet as large as the audio itself -- lands in the samples, where
    # no later high-pass can tell it apart from the signal.
    pos = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    pos -= lo + span / 2
    pos -= np.nanmedian(pos)

    good = np.flatnonzero(np.isfinite(pos))
    if len(good) < len(pos) // 2:
        return None
    pos = np.interp(np.arange(len(pos)), good, pos[good])

    if narrow:
        # Close the window down onto the stroke it just found, and read again.
        #
        # A lane-wide centroid collects the noise of the whole lane -- mostly
        # bare paper, and the further out that paper sits the more leverage its
        # noise has on a first moment -- against the ink of one thin stroke.
        # A stroke fills a tenth of its lane, so nine tenths of what a wide
        # window weighs is paper.
        #
        # Narrowing to a couple of stroke widths is what makes this carrier
        # work at all. At 60/255 of noise the same page reads 0.80 lane-wide
        # and 0.99 narrow.
        #
        # The schedule is wide-with-a-sticky-aim first, then narrow with the
        # aim taken raw. Sticky is dneedle's trick: it steers its needle by a
        # blend of where the geometry says the groove went and where the ink
        # pulls, so no single reading can throw it. Here the prediction that
        # costs nothing is the neighbouring rows -- the stroke moves ~0.2 px a
        # row, so what a dozen rows agree on is where this row's stroke is, and
        # a row whose window landed on the neighbouring lane's stroke, or on
        # bare paper, is outvoted instead of followed. That is worth +0.13 on a
        # bitonal scan, where losing the stroke is how the reader fails.
        #
        # Only the AIM leans on the neighbours; the sample is always the raw
        # centroid of the last pass. Smooth the output and it lags the stroke
        # by as much as the excursion, the window closes on bare paper, and the
        # lane reads tens of pixels wrong. For the same reason the sticky
        # passes are run at nearly the full lane width, where a mis-aimed
        # window still holds the whole stroke and cannot bias what it measures;
        # narrowing to the aperture is left to the last two passes, aimed raw.
        #
        # Six sticky passes: each one is a fixpoint step -- a better aim keeps
        # more of the neighbouring lane's ink out of the window, which makes
        # the next aim better again. It has settled by six, the gain past that
        # is under 0.002.
        centre = (x0 - off) + span / 2 + ramp
        wide = min(4 * narrow, span)
        guess = pos
        for width, lean in [(wide, sticky)] * 6 + [(2 * narrow, 0), (narrow, 0)]:
            aim = box1d(guess, lean, 0) if lean > 1 else guess
            lo = centre + aim - width / 2
            den = cumsum_at(cs, lo + width) - cumsum_at(cs, lo)
            num = cumsum_at(cx, lo + width) - cumsum_at(cx, lo)
            ok = den > 0.05 * np.median(den)  # window that fell on paper
            fine = np.where(ok, num / np.where(ok, den, 1.0), np.nan) - centre
            guess = np.where(np.isfinite(fine), fine, aim)
        pos = guess
    if not absolute:
        return pos if not narrow else pos - np.median(pos)
    # Page columns, in this function's own frame -- rewritten x0, mid-page y,
    # window riding the skew. traces() used to add these back by hand.
    return rows, x0 + span / 2 + ramp + pos


def strip_bend(mean, strips=STRIPS):
    """The page's own drift, a line per strip of rows, joined into one curve.

    A line is fitted in each strip and read off at the strip's middle, and the
    curve runs straight between those points -- so a page that is honestly
    straight comes back straight, and one that bends is followed rather than
    averaged. The ends run out on their own strip's slope instead of flattening
    off, which is worth the four lines: a strip is an eighth of the page, and
    clamping the last half-strip flat loses real drift at both ends.
    """
    edges = np.linspace(0, len(mean), strips + 1).astype(int)
    y = np.arange(len(mean))
    fits = [np.polyfit(np.arange(a, b), mean[a:b], 1)
            for a, b in zip(edges[:-1], edges[1:])]
    at = [(a + b - 1) / 2 for a, b in zip(edges[:-1], edges[1:])]
    xs = [0.0] + at + [len(mean) - 1.0]
    ys = ([float(np.polyval(fits[0], 0))]
          + [float(np.polyval(f, x)) for f, x in zip(fits, at)]
          + [float(np.polyval(fits[-1], len(mean) - 1))])
    return np.interp(y, xs, ys)


def lane_drift(ink, lanes, span, narrow=APERTURE, sticky=STICKY, strips=STRIPS):
    """How far a lane's stroke travels sideways per row: the page's own drift.

    Returned row by row, because a page is not obliged to be a line. A first
    pass down every lane, and the mean of what comes back is the page rather
    than any lane: a lane's own centroid is its audio, and the audio is what
    averaging over the sheet takes out, leaving the one thing every lane
    shares.

    Split out because cut_lanes.traces() aims its aperture with the same
    number, for the same reason, and a second copy of eight lines is free to
    drift from this one the day either changes.

    The pass is aimed by ink_lean() rather than started flat, and that is what
    the fit can and cannot do hanging on. The window follows the stroke as it
    goes down the page, and a first pass with nothing to follow loses it on a
    crooked sheet -- after which every refinement is fitted to a lost stroke.
    Measured on 6570-row sheets, r after the read:

        shear     one flat line     aimed, per strip
         24 px       1.0000              1.0000
         36 px       0.9997              1.0000
         38 px       0.9988              1.0000
         40 px       0.2703              1.0000
         42 px       0.0023             -0.0176

    **The ceiling that used to sit at one pitch is gone.** It was never this
    function's: at 40 px of shear on a 38.7 px pitch the PROFILE that cuts the
    lanes was summed straight down, so a lane at the top of the page landed on
    its neighbour's columns at the bottom and the page cut into twelve lanes
    where eleven were printed. That profile is now summed along the same lean
    this pass is aimed with -- read_tracks.col_profile() -- and the same
    synthetic sheet reads r=1.0000 at 42 px, at 240 px and at 1920 px, which is
    16 degrees and off the platen long before it is off this reader. What is
    left is the aim itself, and it is still the whole difference between 0.27
    and 1.00 below where the cut used to fail.

    The strips are the second half, and they are worth having only from an
    aimed pass. Fitted cold they chase the content the mean did not cancel and
    cost up to 0.07 of r on real scans; fitted on top of an aim they pay,
    though not much. Measured on paper, r against the audio that was printed:

        sheet          one line   aimed, per strip
        sheetB_scan     0.9683         0.9683
        sheetC_scan     0.9712         0.9704
        sheetD_scan1    0.9784         0.9781
        sheetD_scan2    0.9810         0.9814
        sheetD_scan3    0.9757         0.9785
        sheetD_scan4    0.9843         0.9844
        01.png          0.9436         0.9445
        02.png          0.9124         0.9141

    Five up, two down, one level, and the largest move either way is 0.0028 --
    0.13 dB on sheetD_scan3, which is the most bent sheet here. Split between
    the two halves, on the three sheets it was taken apart on: the aim carries
    D3's whole +0.0028 and C's whole -0.0008, the strips carry 02's +0.0019.
    Strips of 1, 4, 8 and 16 land within 0.0002 of each other, so the count is
    not a knob worth turning; 8 is what STRIPS says.

    ponytail: a mean over lanes is only geometry while the content cancels, and
    it cancels as the root of the lane count. On a full sheet of 60 to 240
    lanes the strips follow the page; on a ten-lane sheet they would start
    following the audio, which is why they sit on top of an aimed pass and why
    there are eight of them and not eighty.
    """
    leans = ink_lean(ink)
    aim = min(leans, key=abs) if leans else 0.0
    rows = len(ink)

    flat = [p for p in (lane_centroid(ink, a, b, span, aim, narrow, sticky)
                        for a, b in lanes) if p is not None]
    if not flat:
        raise ValueError("no readable tracks")
    n = min(len(p) for p in flat)
    y = np.arange(n)
    bend = aim * (y - n / 2) + strip_bend(np.mean([p[:n] for p in flat], 0), strips)
    bend -= bend.mean()

    # The strips are free to follow the page; they are not free to walk away
    # from it. Against the same edges that aimed the pass, the honest sheets
    # here sit within 3.8 px and a fit that has lost the sheet lands 45 px out.
    if leans:
        gap = min(abs(np.polyfit(y, bend, 1)[0] - lean) for lean in leans) * rows
        if gap > SKEW_GAP:
            print(f"the drift fitted down this page says "
                  f"{np.polyfit(y, bend, 1)[0] * rows:+.0f} px across it and "
                  f"the ink's own edges say "
                  f"{', '.join(f'{lean * rows:+.0f}' for lean in leans)} -- "
                  f"the fit has left the sheet it was measuring, so the skew "
                  f"is read low or not at all. Rescan it straighter")
    return bend


def trim_paper(ink, frac=0.02):
    """Drop the blank paper around the ink.

    A sheet is a sheet: printed with margins, and scanned with whatever else
    the glass saw. What is left after this should be the ink and nothing else,
    because a lane's height in rows is the sample rate -- carry a centimetre of
    margin along and every lane reads as longer than the second it holds.

    Rows and columns both, because the glass sees things that are neither paper
    nor ink. One scan here came back with the scanner's own black edge in its
    top right corner, 14 columns wide and 26 rows tall. In those rows it is 14
    strong pixels against a 2% threshold of 11, so on rows alone it survives
    and props up 250 rows of blank paper above the ink. Read as columns the
    same blob is 26 pixels of a 6550-row column and goes, and the rows it was
    holding up go with it.
    """
    # Counted as pixels that are actually inked, not as a share of the line's
    # total: sensor noise on bare paper sums to a couple of percent of an
    # inked row across a sheet this wide, which is enough to pass for ink and
    # leave the margins in -- and then a lane is measured as tall as the sheet
    # and the whole read comes out at the wrong rate.
    while True:
        before = ink.shape
        for _ in range(2):   # rows, then columns; two transposes is identity
            floor = np.median(ink[::16, ::16])
            strong = (ink > (float(ink.max()) + floor) / 2).sum(1)
            lit = np.flatnonzero(strong > strong.max() * frac)
            # Kept low on purpose: the blur of print and scan fades the first
            # and last rows of ink into the paper, and those rows are samples.
            # Trimming them as margin costs two rows of every lane and 0.02 of
            # the read.
            if len(lit) > 1:
                ink = ink[lit[0]:lit[-1] + 1]
            ink = ink.T
        # Until it stops shrinking: the corner blob holds its own rows in on
        # the pass that sees it as rows, and only the next pass round can drop
        # them once the columns carrying it have gone.
        if ink.shape == before:
            return ink


def read_curves_page(ink, pitch=None, sr=None, narrow=None,
                     sticky=STICKY, estimator=None):
    """Read a page of curves: the ink's centroid in each lane, per row.

    `narrow` defaults to APERTURE scaled by the sheet's own pitch -- see the
    window rule where it is applied below. A number given here is taken as
    pixels and left alone, which is what `read --aperture` is for.

    The centroid needs no baseline and no threshold -- it is measured against
    the lane's own centre -- and it is blind to how much ink landed, so print
    density, exposure and gamma drop out of the answer entirely.

    Skew cannot be read by cross-correlating strips of the page against the
    top strip, which is the obvious way and works only while something in the
    profile is nailed down. A curve's position is not: the estimator returns
    the audio's own slow wander as if it were geometry, and reports -6.85 px of
    "skew" down a page drawn with none.

    What is common to every lane is the skew and what is not is the audio,
    since each lane holds a different second. So the skew is the mean of the
    lanes' own centroids: content cancels as the square root of the lane count,
    geometry survives. First pass assumes none and measures it, second pass
    rides it, and that second pass is most of the work in a read.

    The clock lanes could hand this over as a measurement instead, and they are
    not asked to. Their content is known, so what is left of their trace is the
    page and nothing else -- but on the crooked sheet their skew lands on
    -0.002686 px/row against this estimate's -0.002684. Four figures, for a
    measurement this one makes for free, and a grid laid between them scores
    0.12 dB worse than the segmentation below. There is nothing there to win.

    What this estimate is worth is a property of the sheet, not of the reader.
    Leaving the skew alone costs 2.34 dB on the crooked sheet, which lay 17.7
    px crooked, and 0.09 dB on the rippled sheet, which lay 4.9 px crooked. It
    is worth as much skew as there is, which is not knowable until the sheet is
    on the platen -- so it is kept for the crooked sheet, not for the average
    one. It costs nothing and needs no mark on the page.
    """
    ink = trim_paper(ink)
    lanes = find_tracks(ink, pitch)
    span = int(np.median([b - a for a, b in lanes]))
    # The window follows the pitch. APERTURE is 2 x STROKE, tuned at 38.7 px,
    # and what it has to hold is the smear one row of a moving stroke leaves --
    # which scales with the lane, because the excursion does. Measured on three
    # geometries: the optimum is 4 px at pitch 18.4, 8 at 38.7 and 16-24 at
    # 77.4, and reading the 77.4 sheet at 8 costs 6.0 dB. On the sheets that
    # were already here it changes nothing worth the name -- 37.4 px of pitch
    # asks for 7.7 px of window instead of 8, and every score holds to four
    # figures.
    if narrow is None:
        narrow = APERTURE * span / PITCH

    for said in (odd_lane(lanes), steepened(ink, lanes)):
        if said:
            print(said)

    drift = lane_drift(ink, lanes, span, narrow, sticky)

    # `estimator` swaps out what a lane's sample IS, and nothing else: the
    # grid, the drift and the assembly below are the same either way, which is
    # the only way two readouts can be compared on one scan. The drift pass
    # above is deliberately NOT swapped -- it is the page's geometry, every
    # lane agrees about it, and holding it fixed leaves one variable.
    tracks = [(estimator or lane_centroid)(ink, b[0], b[1], span, drift,
                                           narrow, sticky)
              for b in lanes]
    live = [len(t) for t in tracks if t is not None]
    if not live:
        raise ValueError("no readable tracks")
    if sr is None:
        sr = int(np.median(live))

    # A lane that is not there is still a second of the recording. Dropped
    # instead, the file comes back a second short and everything past the hole
    # plays a second early -- which scores as noise from that point on while
    # still sounding like the recording, because a second is well under the
    # ear's patience for a join. Measured on a sheet of 20 lanes with the ninth
    # wiped: read as 19 lanes it correlates 1.00 for eight seconds and then
    # 0.04, -0.04, 0.06, -0.04 to the end of the page.
    #
    # Two ways to be missing, and both land here. A lane the segmentation never
    # found leaves a gap in the grid, and the grid is regular, so the gap is
    # its own measure: how far this lane starts from the last one, over the
    # pitch. A lane that was found and could not be centred comes back None.
    # Either way the silence goes in where the second belongs, and what is
    # returned is a count of SECONDS rather than of lanes read, so the clocks
    # at the ends are still where the retiming expects them.
    song, holes = [], []
    for i, ((x0, _), t) in enumerate(zip(lanes, tracks)):
        for _ in range(int(round((x0 - lanes[i - 1][0]) / span)) - 1 if i else 0):
            holes.append(len(song))
            song.append(np.zeros(sr))
        if t is None:
            holes.append(len(song))
            song.append(np.zeros(sr))
        else:
            song.append(resample(t, sr))
    if holes:
        shown = ", ".join(str(h + 1) for h in holes[:8])
        print(f"second{'s' if len(holes) != 1 else ''} {shown}"
              f"{' ...' if len(holes) > 8 else ''} of {len(song)}: no lane on "
              f"the sheet to read there -- silence in their place, so the "
              f"seconds after them stay where they belong")
    return np.concatenate(song), sr, len(song)


SHARP = 0.95    # of the printed stroke width; below this the scan was steepened


def stroke_fwhm(ink, lanes, rows=None, look=12, sample=12):
    """Width of the stroke where it is half its own height, in pixels.

    Rows are averaged about each row's OWN peak rather than about the lane's
    centre, because the stroke moves: averaged about a fixed column it would
    come back as wide as the excursion and measure the audio instead of the
    ink. Lanes are sampled across the page so one bad one cannot set it.
    """
    if rows is None:
        rows = (len(ink) // 4, len(ink) * 3 // 4)
    step = max(1, (len(lanes) - 4) // sample)
    prof = []
    for t in range(2, len(lanes) - 2, step):
        x0, x1 = lanes[t]
        crop = ink[rows[0]:rows[1], x0:x1].astype(np.float64)
        if crop.shape[1] < 2 * look + 1:
            continue
        for i, k in enumerate(crop.argmax(1)):
            if look <= k < crop.shape[1] - look:
                prof.append(crop[i, k - look:k + look + 1])
    if len(prof) < 100:
        return None
    p = np.mean(prof, 0)
    p = p - np.median(p)                       # bare paper to zero
    if p.max() <= 0:
        return None
    half = p.max() / 2
    on = np.flatnonzero(p >= half)
    lo, hi = on[0], on[-1]
    if lo == 0 or hi == len(p) - 1:
        return None
    left = lo - (p[lo] - half) / max(p[lo] - p[lo - 1], 1e-9)
    right = hi + (p[hi] - half) / max(p[hi] - p[hi + 1], 1e-9)
    return float(right - left)


def steepened(ink, lanes, stroke=STROKE, limit=SHARP):
    """Has this scan been sharpened, or its response steepened? -> note or None.

    The sheet prints its own ruler. A stroke leaves the printer STROKE pixels
    wide and can only come back WIDER: paper spreads ink and glass spreads
    light, and neither has ever narrowed a line. So a stroke that measures
    narrower than it was printed is not a property of the paper -- it is the
    scanner steepening its response, pushing the partial-coverage pixels at the
    stroke's edges towards black and towards white.

    Those pixels are the whole of the sub-pixel position, so this is the same
    damage a white point does, and it is worth a separate check because
    `retouched` cannot see it: a curve applied inside the scanner is applied
    BEFORE the value is quantised, so it leaves no comb of empty bins. Measured
    on one such scan: 1 empty bin, and a stroke 3.2 px wide out of 4.

    Measured widths, printed stroke 4 px: 4.00 on a sheet straight out of
    `print`, 4.32 to 4.59 on six scans that read between 0.956 and 0.979, and
    3.21 to 3.30 on the two that read 0.10 and 0.64. Nothing lands between 4.00
    and 4.32, which is where the line goes.

    The width expected is STROKE flat, not scaled by anything. The stroke is
    the one thing here that does not follow the pitch: `--pitch` moves the
    lane's walls and so its excursion, `--dpi` moves the sheet's rate, and
    neither touches how wide the line is drawn. Scaling it by the pitch is the
    obvious mistake and it accuses the widest sheet in this directory of a
    defect it does not have -- sheetC_scan, pitch 76.9, reads a perfectly
    ordinary 4.4 px.

    What this does assume is a scan at the printed dpi, which the README asks
    for anyway. Scanned at half of it a 4 px stroke arrives 2 px wide and this
    will say so, which is not the worst thing wrong with such a scan.
    """
    want = stroke
    got = stroke_fwhm(ink, lanes)
    if got is None or got >= want * limit:
        return None
    return (f"the stroke measures {got:.1f} px where it was printed "
            f"{want:.1f}, and ink on paper only ever spreads. Something in the "
            f"scan is steepening the edges -- sharpening, or the scanner's own "
            f"contrast -- and the pixels it is squaring off are the sub-pixel "
            f"position itself. Rescan with every adjustment off, or on another "
            f"scanner")


def retouched(ink, comb=COMB):
    """Has this scan been through a tone curve? -> a note to print, or None.

    A scanner's own output fills its histogram: every level between paper and
    ink occurs, because the ink's edges land on every fraction of a pixel.
    Stretch that with a levels tool and the same values are spread over a wider
    range, leaving a comb of empty bins behind. Measured: 0 empty bins on a
    sheet straight out of render_page, 4 on both raw scans in this directory,
    and 128 to 133 on all four that have been retouched.

    It cannot say whether the retouching HURT, and that is not a failure of
    this check -- three attempts at measuring the harm from the image alone all
    came to nothing, and one of them ranked the ruined scan cleaner than the
    intact ones. What is known is the asymmetry. A white point that reaches
    into the ink costs a curve read 8.6 dB and cannot be undone, a rescan costs
    a minute, and a levels tool leaves this trace whether or not it did damage.
    So the note names what was found and what to do about it, and does not
    accuse the scan of being broken.

    The crooked sheet has been through a levels tool and reads fine. This will
    say so about it, and that is the right answer: nothing here can tell it
    apart from the retouched scan that lost 8.6 dB.
    """
    top = int(ink.max())
    if top < 2:
        return None
    empty = int((np.bincount(ink.ravel(), minlength=256)[:top + 1] == 0).sum())
    if empty < comb:
        return None
    return (f"levels have been adjusted after scanning ({empty} empty "
            f"histogram bins). That may have cost nothing, or it may have cost "
            f"8 dB -- nothing in the image says which. If this read is "
            f"disappointing, rescan with every adjustment off")


# ------------------------------------------------------------------- printing

def lowpass(x, ratio, cycles=DEC_TAPS):
    """Windowed sinc cutting at `ratio` of Nyquist, as long as the ratio needs.

    The length is the whole of it. A kernel is worth the number of cycles of
    its own cutoff it holds, not the number of samples: 33 samples is eight
    cycles at half Nyquist and two and a half at a sixth of it. Fixed at 33 and
    asked to take 44.1 kHz down to 6780, it drops 3.1 dB at 3 kHz -- worse than
    the box it replaced, which dropped 1.8. Scaled, the same kernel is flat to
    0.2 dB there, and the whole of a 5.3 million sample file costs 0.32 s
    against 0.16.
    """
    t = int(np.ceil(cycles / ratio))
    h = np.sinc(np.arange(-t, t + 1) * ratio) * np.hanning(2 * t + 1)
    return np.convolve(x, h / h.sum(), "same")


def to_rate(signal, sr, rate):
    """Resample, windowed-sinc first when thinning.

    The sheet's Nyquist is half its height in rows, and anything above it
    folds down into the audio rather than being lost quietly. The same kernel
    thin() uses -- a box here used to cost 11.5 dB of a digital sheet against
    the sinc's 3, all of it passband droop.
    """
    if rate == sr:
        return signal
    n = int(round(len(signal) * rate / sr))
    if n < 2:
        return np.asarray(signal)[:n]
    if rate < sr:
        signal = lowpass(signal, rate / sr)
    return np.interp(np.linspace(0, len(signal) - 1, n),
                     np.arange(len(signal)), signal)


def paper_mm(paper):
    """Sheet size in mm, by name or as WxH."""
    if paper in PAPER:
        return PAPER[paper]
    try:
        w, h = (float(v) for v in paper.lower().split("x"))
    except ValueError:
        raise SystemExit(f"paper: give one of {', '.join(PAPER)} or WxH in mm")
    return w, h


def mm_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))


def sheet_px(paper, dpi, margin):
    """Printable area in pixels: the paper less the margin the printer eats."""
    w, h = paper_mm(paper)
    return mm_px(w - 2 * margin, dpi), mm_px(h - 2 * margin, dpi)


def lanes_that_fit(width, pitch, edge=None):
    """Seconds a sheet holds: its width less the blank edges."""
    edge = pitch if edge is None else edge
    return int((width - 2 * edge) // pitch)


def pilot_lane(rate, per=PILOT):
    """One lane of a plain tone, to be read back as the sheet's own clock.

    A whole number of cycles in the lane, so the lane begins and ends on the
    same phase and the reader can find the tone in a single FFT bin without
    being told which.

    Half the lane's swing, not all of it: the phase is measured against the
    scan's own noise, so more amplitude is more precision -- but a full-swing
    tone crosses the lane in a couple of rows, which is the one thing print and
    scan render worst, and the precision is wanted exactly there.
    """
    cycles = max(1, round(rate / per))
    return PILOT_AMP * np.sin(2 * np.pi * cycles * np.arange(rate) / rate)


def do_print(args):
    from PIL import Image

    signal, sr = read_wav(args.audio)
    # Scaled by the same high percentile write_wav() reads back with, not by
    # the peak. The excursion IS the carrier -- a sample is where the stroke
    # sits across its lane -- so a source whose peak is one click prints every
    # other sample as a fraction of the lane it could have had, and the sheet
    # comes back quiet for a reason no scan of it can show. Measured through
    # the scan model (box blur 5, noise 10/255), against audio carrying one
    # click at full scale over music at 0.15: r 0.8953 by the peak, 0.9317 by
    # the percentile, which is 2.1 dB for a defect the eye cannot see on the
    # sheet. Clean and full-swing audio move by 0.0004, downwards, which is
    # the clipping this costs.
    #
    # HEADROOM is the read side's own number and is measured there: a healthy
    # file's peak sits 1.19 to 1.46 above its 99.9th percentile, so 1.3 clips
    # 0.00% of music and only the click.
    scale = np.percentile(np.abs(signal), 99.9) * HEADROOM
    signal = np.clip(signal / scale, -1.0, 1.0) if scale else signal * 0.0
    over = float(np.mean(np.abs(signal) >= 1.0))
    if over > 0.0005:
        # Audio compressed hard enough that its 99.9th percentile is nearly its
        # peak is the one case this rule costs something, so it is said. Below
        # that it is the click the rule is for.
        print(f"NOTE: {over:.2%} of the samples print at the edge of their "
              f"lane and are flattened there -- this audio is compressed hard "
              f"enough that its 99.9th percentile is nearly its peak")
    width_px, rate = sheet_px(args.paper, args.dpi, args.margin_mm)
    pitch = args.pitch
    # Rows per sample. Everything below counts in ROWS -- a lane is `rate` rows
    # whatever happens -- and the audio is simply resampled `rows` times as
    # dense, so a lane holds 1/rows of a second and the ink moves half as far
    # per row at 2. Nothing else in the layout knows about it.
    rows = int(getattr(args, "rows", 1) or 1)
    # A lane narrower than the stroke plus its clearance has no room left
    # to swing in: amp goes to zero and then negative, which prints the
    # sound inverted and unreadable rather than failing.
    if (pitch - STROKE) / 2 - MARGIN < 1:
        raise SystemExit(f"pitch: {pitch:g} px leaves no excursion; "
                         f"needs more than {STROKE + 2 * MARGIN + 2:g}")
    # Not on the command line: the project's own table prices the clocks at
    # 1.6% of the sheet for 1.4-4.9 dB, so there is no sane reason to decline
    # them. The selftest still prints clockless sheets through here.
    pilot = bool(getattr(args, "pilot", True) and PILOT)
    paper_w, paper_h = paper_mm(args.paper)
    sheet_w, sheet_h = mm_px(paper_w, args.dpi), mm_px(paper_h, args.dpi)
    signal = to_rate(signal, sr, rate)[int(args.start * rate):]
    if rows > 1:
        # Stretched here rather than by asking to_rate for `rate * rows`: its
        # grid runs endpoint to endpoint, which is a rate of (2n-1)/(n-1) and
        # not 2, and half a row of drift across a sheet is not something the
        # reader's exact decimation can put back. This grid puts every original
        # sample ON a row, so taking every `rows`-th row returns it untouched.
        signal = np.interp(np.arange(len(signal) * rows) / rows,
                           np.arange(len(signal)), signal)
    # The sheet's own margin is the blank the outermost lane needs, so only
    # what the margin falls short of a pitch has to be drawn as well. At any
    # sane margin that is nothing, and the two pitches it used to cost come
    # back as two more seconds on the sheet.
    edge = max(0.0, pitch - mm_px(args.margin_mm, args.dpi))
    fits = lanes_that_fit(width_px, pitch, edge)
    pilot = pilot and fits > 3
    fits -= 2 * pilot            # a clock takes a lane like any other second
    per_sheet = fits
    if per_sheet < 1 or not len(signal):
        raise SystemExit("nothing to print")

    if sr != rate * rows:
        # The kernel is as long as the ratio needs (lowpass), so this is no
        # longer the box that cost a couple of dB: 44.1 kHz down to 6780 comes
        # out flat to 0.2 dB against a 513-tap resample, and 0.8 dB down at the
        # very top. Said anyway, because nothing in prep*.bat meets it -- they
        # hand over `rate -v 6780` already -- and a resample the reader cannot
        # see is worth one line on the way past.
        print(f"  NOTE: {args.audio} is {sr} Hz and this sheet holds "
              f"{rate * rows}; resampled here through a "
              f"{2 * int(np.ceil(DEC_TAPS * sr / (rate * rows))) + 1}-tap sinc. "
              f"`sox in.wav -r {rate * rows} out.wav rate -v {rate * rows}` "
              f"is the same job done by a resampler")
    lanes_wanted = int(np.ceil(len(signal) / rate))
    sheets = int(np.ceil(lanes_wanted / per_sheet))
    base = (args.out or args.audio.rsplit(".", 1)[0]).removesuffix(".png")
    print(f"{args.audio}: {lanes_wanted / rows:g} s at {rate * rows} Hz, "
          f"{per_sheet / rows:g} per sheet "
          f"-> {sheets} sheet{'s' if sheets != 1 else ''}")
    print(f"  {args.paper} {paper_w:g}x{paper_h:g} mm = {sheet_w}x{sheet_h} px "
          f"at {args.dpi} dpi, margin {args.margin_mm:g} mm, "
          f"pitch {pitch:.1f} px = {pitch * 25.4 / args.dpi:.2f} mm, "
          f"clock {'yes' if pilot else 'no'}"
          + (f", {rows} rows a sample" if rows > 1 else ""))

    # STROKE, MARGIN and APERTURE are in PIXELS, and --dpi does not touch them:
    # a sheet printed at another resolution keeps its pixel geometry and
    # changes its physical one. The only geometry that has been on paper here
    # is 600 dpi -- an 0.17 mm stroke in a 1.64 mm lane, which came back off a
    # flatbed at 0.78 to 0.94. Half that stroke is not a smaller version of the
    # same format: an inkjet lays 0.085 mm down intermittently and a scanner
    # reads what is left as grey. Nothing downstream says so -- a sheet at 1200
    # dpi that never leaves the computer still reads back at 0.9991, which is
    # the whole reason this is printed here and not left to the bench.
    stroke_mm = STROKE * 25.4 / args.dpi
    if stroke_mm < 0.15:
        scale = args.dpi / DPI
        print(f"  NOTE: {stroke_mm:.3f} mm of stroke, against the 0.17 mm this "
              f"format has been printed and scanned at. STROKE, MARGIN and "
              f"APERTURE are pixels and do not follow --dpi; STROKE "
              f"{STROKE * scale:.0f} and --pitch {pitch * scale:.0f} would keep "
              f"the sheet the size it is at {DPI} dpi")

    for i in range(sheets):
        chunk = signal[i * per_sheet * rate:(i + 1) * per_sheet * rate]
        lanes = int(np.ceil(len(chunk) / rate))
        if lanes < 2:
            # A sheet is segmented by its own periodicity and one lane has
            # none: the estimator searches from two periods up, so a single
            # lane is not an answer it can give. Silence is cheaper than a
            # sheet nothing can read.
            lanes = min(2, fits)
        # A lane is a whole second whatever happens, so the tail is padded with
        # silence -- a straight stroke down the middle of its lane.
        chunk = np.pad(chunk, (0, max(0, lanes * rate - len(chunk))))
        held = lanes             # seconds of sound; the clocks are not seconds
        if pilot:                # one at each end, bracketing the field
            # In SAMPLES the clocks are always PILOT and PILOT_END; on paper
            # they come out `rows` times longer, and that is the only place a
            # sheet says how fast it was printed.
            chunk = np.concatenate([pilot_lane(rate, PILOT * rows), chunk,
                                    pilot_lane(rate, PILOT_END * rows)])
            lanes += 2

        edges, amp = lay_out(chunk, rate, lanes, pitch, 0.0, edge=edge)
        width = int(np.ceil(max(r.max() for _, r in edges) + edge))
        # The file is the sheet, not the ink on it: the lanes are laid into a
        # page of the full paper size, centred, so the margin is white paper in
        # the image itself. Printed at 100% there is then nothing to line up by
        # hand, and nothing to wonder about when the ink is narrower than the
        # sheet -- a nine-second tail is still a sheet of A4.
        page = np.zeros((sheet_h, sheet_w), np.float32)
        y, x = (sheet_h - rate) // 2, (sheet_w - width) // 2
        page[y:y + rate, x:x + width] = render_page(edges, pitch, width)
        out = f"{base}_{i + 1:02d}.png" if sheets > 1 else f"{base}.png"
        Image.fromarray(255 - np.round(page * 255).astype(np.uint8)).save(
            out, dpi=(args.dpi, args.dpi))
        # The margin asked for is a minimum: the lanes are a whole number, the
        # sheet is not, and the remainder is split between the two sides.
        print(f"  {out}: {held} lane{'s' if held != 1 else ''}, "
              f"seconds {args.start + i * per_sheet / rows:g}-"
              f"{args.start + (i * per_sheet + held) / rows:g}, "
              f"ink {width}x{rate} px = "
              f"{width * 25.4 / args.dpi:.0f}x{rate * 25.4 / args.dpi:.0f} mm, "
              f"margins {x * 25.4 / args.dpi:.1f} mm sides, "
              f"{y * 25.4 / args.dpi:.1f} mm ends")


# -------------------------------------------------------------------- reading

def despeckle(x, thresh=DESPECKLE, reach=10):
    """Flatten samples that leave the trace and come straight back.

    A speck of dust inside the aperture, or a dropout in the scan, pulls the
    measurement for as long as it covers: a step out and a step back, a few
    samples apart. This is the impulse rejection used in optical-sound
    restoration, which was written for the same material -- the step is taken
    out of the DERIVATIVE and the trace re-integrated, so a burst is flattened
    rather than merely notched, and the signal either side of it keeps its own
    level.

    The threshold is a multiple of the MEDIAN step rather than of the largest:
    the largest step is set by the worst artefact on the page, which is the one
    thing the threshold exists to catch.

    Off by default, and that is a measurement, not caution. Against the audio
    that was printed, on a scan with specks of solid ink pasted into it:

        blots on the sheet     off     at 20
        none                  0.811    0.753
        200 of r=4px          0.786    0.742
        2000 of r=4px         0.559    0.583
        2000 of r=8px         0.323    0.476

    On a clean or lightly marked page it only takes signal -- a printed stroke
    at a real transient steps as hard as a blot does, and nothing in the trace
    tells them apart. Swept over the crooked sheet, an ordinary scan with no
    blots on
    it, every setting is worse than off and the cheapest of them is ruinous:

        off    2      3      5      8      12
        0.781  0.122  0.290  0.474  0.614  0.684

    It pays on a page that is genuinely filthy, which is what the flag is for,
    and it is off by default because most pages are not. What looked like impulse noise on these scans was not:
    it was every lane being read from the page's rows instead of its own.
    """
    if not thresh:
        return x
    d = np.diff(x)
    lim = thresh * np.median(np.abs(d))
    if not lim:
        # More than half the read is flat: a page lost whole, a tail of blank
        # lanes, a genuinely silent passage. The median step is then zero, and
        # a threshold of zero makes an outlier of every sample there is -- the
        # flag would flatten the whole file, which is the opposite of what it
        # is for. Nothing to measure against, so nothing is taken out.
        return x
    for i in np.flatnonzero(np.abs(d) > lim):
        j = i + 1
        while j < len(d) and j - i <= reach and abs(d[j]) > lim:
            j += 1          # ride out the burst, up to `reach` samples of it
        d[i] = (d[i - 1] + d[j]) / 2 if 0 < i and j < len(d) else 0.0
    return np.concatenate(([x[0]], x[0] + np.cumsum(d)))

def pilot_bin(lane, slack=PILOT_SLACK):
    """The clock's bin, printed rate and rows per sample, or None if not a clock.

    -> (k, per, rows). `rows` is the sheet's own speed, and this is the only
    place it is measured: nothing else on a sheet says how many rows a sample
    was given.

    Two tests, and the second one is the one that works. A tone is a single bin
    and a second of music is not -- but the trace of a curve lane is not audio,
    it is a position, and its slow wander dominates everything: on an early
    test scan the peak bin of every music lane is bin ONE, holding up to 73% of
    the trace on its own. An energy test alone passed all 41 lanes sampled as
    clocks.

    So the bin has to be where the format put it: PILOT or PILOT_END samples
    to the cycle, times ROWS rows to the sample -- 20, 32, 40 or 64 rows,
    whichever is nearer in log. The ladder is what sets the slack: its closest
    pair is 1.25 apart, so PILOT_SLACK is half of that and can be no more. It
    is still four times the worst scale error a scan here has shown and two
    orders of magnitude clear of the wander. Bin 1 against a clock at bin 206
    is not a close call, and neither is telling the two clocks apart: they sit
    a ratio of 1.6 from each other and a scan's scale error is under a percent.
    Which rate this lane carries is what pilot_retime reads the sheet's
    orientation from; which rung it sits on is the sheet's speed.

    A line comes off first, not just the mean. On a sheet that never left the
    computer the wander is flat enough that the mean is enough; on a real scan
    the page is skewed, and the skew puts a ramp down the clock's trace -- 18
    px of it on the crooked sheet, against a tone of 7 px rms. A ramp is not a
    tone, but its energy is all in the low bins, so the peak lands on bin 1 and
    the energy test fails, so every clock on the crooked sheet read as music
    and the sheet came back untimed. The skew is geometry, never signal, and
    comes off here.
    """
    n = len(lane)
    t = np.arange(n)
    P = np.abs(np.fft.rfft(lane - np.polyval(np.polyfit(t, lane, 1), t))) ** 2
    k = int(np.argmax(P[1:])) + 1
    if not PILOT:
        return None
    per, rows = min(((p, m) for p in (PILOT, PILOT_END) for m in ROWS),
                    key=lambda c: abs(np.log(n / (k * c[0] * c[1]))))
    if not 1 / slack <= n / (k * per * rows) <= slack:
        return None
    return ((k, per, rows) if P[max(k - 2, 0):k + 3].sum() >= 0.5 * P.sum()
            else None)


def pilot_timebase(lane, periods=1):
    """Timing error per sample, in samples, from a lane carrying one tone.

    -> (the error, the clock's printed rate, the sheet's rows per sample),
    or None if not a clock.

    A scanner does not advance one row per row. Its carriage ripples, and every
    lane on the sheet is sampled on the same rippling clock: measured against
    the audio that was printed, the lanes of a scan agree with each other about
    the ripple at 0.80, which is what a shared time base looks like and audio
    does not.

    How much of it there is varies by the day. The crooked sheet's two clocks
    read 1.01 and 1.42 samples rms and taking it out was worth 1.44 dB; the
    rippled sheet read 1.38 and 9 samples peak to peak, and was worth 4.88 dB.

    Nothing on a finished sheet without clocks can measure it. The ripple is
    purely vertical, so it leaves no horizontal signature: across a scan the
    printed field's own width holds to 0.020%, which is ten times too steady to
    explain a ripple of several samples, and the ink per row moves with the
    audio rather than the carriage. A sheet that carries no clock cannot be
    retimed, and this returns None rather than guessing.

    Do not size this correction with a stand-in clock -- a music lane of known
    content used as if it were one. That measures the ripple against audio the
    reader already has, where a printed clock is measured against nothing and
    puts its own noise into the correction. Tried here, the stand-in promised
    two and a half times what the printed clock delivered.

    Quadrature demodulation, the ordinary way to read wow off a pilot tone:
    the tone is shifted to DC, smoothed over whole periods -- which nulls the
    image at twice the tone exactly -- and what is left of the phase is the
    clock's own error.

    One period of smoothing, not several. The smoothing is a boxcar over the
    thing being measured, and the ripple is not slow: on a scan here it runs
    to 30 cycles down the sheet, where four periods of boxcar already cost
    half the amplitude. One period nulls the image just as exactly and takes 3%
    off instead.
    """
    got = pilot_bin(lane)
    if got is None:
        return None                      # no single tone: this is audio
    k, per, rows = got
    # The same line pilot_bin takes off to FIND the clock has to come off to
    # MEASURE it. A skew ramp is not nulled by the boxcar below -- it lands
    # beside the null, not on it -- and it drags the answer: against a ripple
    # drawn at 30 cycles down the sheet, the 32 clock recovers it at 0.996
    # detrended and 0.928 with the crooked sheet's 18 px of ramp left in.
    t = np.arange(len(lane))
    x = lane - np.polyval(np.polyfit(t, lane, 1), t)
    n = len(x)
    z = x * np.exp(-2j * np.pi * k * np.arange(n) / n)
    m = max(2, int(round(periods * n / k)))
    z = box1d(z.real, m, 0) + 1j * box1d(z.imag, m, 0)
    j = np.unwrap(np.angle(z)) * n / (2 * np.pi * k)
    return j - j.mean(), per, rows       # a constant is a delay, not a ripple


def retime(song, sr, j, j_end=None):
    """Resample every lane off the clock the sheet came with.

    With `j_end`, the two clocks are read as the ends of a straight line and
    each lane is retimed off its own point on it, rather than off the average
    of the two. That is not a refinement of the average, it is the answer to
    what the two clocks disagree about: on the crooked sheet they correlate at
    0.61 raw and 0.82 once each one's own line comes off, so the wiggle is
    shared and the trend is not -- -1.24 samples across the left clock's lane
    against +1.63 across the right. A carriage rippling vertically cannot do
    that; the sheet's own geometry can, and it varies across the page by
    definition.

    Worth 0.14 dB there, and it costs nothing: the clocks are lanes 0 and
    n - 1, so the line between them is already measured.
    """
    t = np.arange(sr)
    nl = len(song) // sr
    out = []
    for k in range(nl):
        jk = j if j_end is None or nl < 2 else (1 - k / (nl - 1)) * j + (k / (nl - 1)) * j_end
        out.append(np.interp(np.clip(t - jk, 0, sr - 1), t, song[k * sr:(k + 1) * sr]))
    return np.concatenate(out)


def thin(song, sr, rows):
    """One sample per `rows` printed rows: filtered, then lane by lane.

    The filter is the whole point of printing more than one row to a sample,
    not a detail of the thinning: at 2 rows the audio uses the bottom half of
    the sheet's band and the top half is nothing but scan, and dropping every
    other row without filtering folds all of that straight down into the audio
    -- which is exactly the noise the format just paid half a sheet to get
    away from. The same windowed sinc to_rate() uses when a wav sits above
    the sheet's rate.

    Filtered across the whole read and sliced per lane afterwards, so a lane
    holds a whole number of samples whatever rate the scan came back at. The
    filter DOES reach across a lane join, by about eight samples either side,
    and that is the point rather than a leak: a join is continuous audio and
    not a boundary, so filtering each lane on its own would invent an edge the
    paper does not have and taper the last samples of every second towards
    zero. Only the two ends of the whole read see the filter's own edge.
    Measured, the choice is free either way -- per lane and across the join
    both score 25.04 dB on a digital two-row sheet -- so it is made on what is
    right rather than on decibels.
    """
    if rows < 2:
        return song
    # `keep` is a floor, and on an odd scan that throws away half a sample of
    # TIME from every lane. Kept at the sheet's own rate the read then runs
    # short by half a sample a lane -- 59 samples across a sheet of 118, which
    # is inaudible and fatal to any score taken at one lag. The samples are
    # right; the rate is what has to follow them, and do_read takes it from
    # here as `(sr // rows) * rows`.
    filtered = lowpass(song, 1 / rows)
    keep = sr // rows
    return np.concatenate([filtered[k * sr:k * sr + keep * rows:rows]
                           for k in range(len(song) // sr)])


def clock_order(rates, verb):
    """Which way up the sheet lay, from what its two end clocks read back.

    -> (turned, what to say about it, or None). `rates` is the printed rate of
    the head clock and of the tail one, either of them None where that end did
    not read as a clock; `verb` is what the caller does to a sheet, "read" or
    "cut", since the answer to a sheet that cannot say is to do it again
    turned.

    One rule, one place, because there are two readers and they turn different
    things: paper_sound turns the AUDIO -- reversing the signal and its sign IS
    the 180 degree turn, lane order, time and excursion in one stroke -- and
    cut_lanes turns the IMAGE and segments it again. Two mechanisms that must
    never disagree about WHETHER the sheet was turned, because a sheet played
    backwards is the one fault on this format that every other measurement
    reports as fine.

    Turned on TWO clocks reading in the wrong order and on nothing less. One
    clock is not evidence, however tempting it looks: pilot_bin takes any lane
    whose trace is a tone within an octave of a clock's rate, and an audio lane
    can be one -- a clockless sheet here reads [None, 20], its last second
    being a 24 Hz sine in a 400-sample lane. Acting on that alone turns a whole
    sheet back to front on the strength of one held note.

    Two clocks at the SAME rate cannot say either -- that sheet was printed
    before the rates differed -- and that is worth saying out loud rather than
    treating as a quiet no. 04.png here is such a sheet: it reads 0.0029
    against the audio that went onto it the way it was scanned and 0.4706
    turned, and neither number is visible from inside.
    """
    if list(rates) == [PILOT_END, PILOT]:
        return True, None
    if rates[0] is not None and rates[0] == rates[1]:
        return False, (f"both clocks are the {rates[0]}-sample one, so this "
                       f"sheet cannot say which way up it lay -- it was "
                       f"printed before the two rates differed. If it plays "
                       f"backwards, {verb} it again with --upside-down")
    return False, None


def pilot_retime(song, sr, n, pilot=True, rows=None):
    """Retime a read off its clock lanes and cut them off the ends.

    -> (song, lanes, per-clock timebases, turned, rows per sample). `rows`
    overrides what the clocks say about the sheet's speed, and is the only way
    to read a thin sheet whose clocks are gone -- nothing else on a sheet says
    how many rows a sample got. Anything that scores or
    plays a read goes through here, not just do_read: a clocked sheet read
    without it comes back a whole lane late, with a clock in place of the
    first second, and correlates at zero against the file that was printed.

    The two clocks are printed at different rates -- PILOT samples a cycle on
    the left, PILOT_END on the right -- because in every other respect a sheet
    is symmetric under a 180 degree turn, and which way up it went on the
    glass used to be the one thing it could not say. Reading them back in the
    wrong order means the sheet lay upside down, and the turn is applied to
    the read itself: reversing the signal and its sign IS the 180 degree turn,
    lane order, time and excursion in one stroke. `turned` says it happened.
    Two clocks at the same rate -- a sheet printed before they differed -- say
    nothing either way, and the read is left alone.

    Two clocks are retimed against as the two ends of a line, not averaged --
    see retime(). Averaging assumes they saw the same thing and they do not:
    on the crooked sheet it scores 0.7761 against 0.7789 for the right-hand
    clock used alone, which is the average being dragged by an error it was
    supposed to cancel. Riding the line between them scores 0.7810 and needs no
    choice made about which end to trust.
    """
    if not pilot or len(song) < 3 * sr:
        return thin(song, sr, rows or 1), n, [], False, rows or 1
    def ends(song):
        last = len(song) // sr - 1
        got = [(k, pilot_timebase(song[k * sr:(k + 1) * sr])) for k in (0, last)]
        return [(k == last, *g) for k, g in got if g is not None]
    got = ends(song)
    rates = [None, None]
    for last, _, per, _ in got:
        rates[bool(last)] = per
    turned, said = clock_order(rates, "read")
    if said:
        print(said)
    if turned:
        song = -song[::-1]
        got = ends(song)                 # the same clocks, in the sheet's frame
    js = [j for _, j, _, _ in got]
    if not js:
        return thin(song, sr, rows or 1), n, js, turned, rows or 1
    song = retime(song, sr, js[0], js[1] if len(js) > 1 else None)
    # Cut a clock off each end that carried one, which is not always the head:
    # a sheet whose left clock is damaged, or whose grid overran on the left,
    # reads its clock at the far end -- 04.png here finds one clock and four
    # lanes of nothing past it. Cutting the head regardless throws away a
    # second of audio and leaves the clock in the read as a second of tone.
    song = song[sr * (not got[0][0]):len(song) - sr * got[-1][0]]
    # The clocks are read at full height and they are what says how many rows a
    # sample got, so the thinning is here, after them.
    #
    # It takes BOTH clocks agreeing. The two mistakes are not each other's
    # mirror: thinning a sheet that was not printed thin throws away the top
    # half of its band -- 32.64 dB to 12.49 measured -- and the wav will not
    # give it back, while reading a thin sheet as a plain one costs a `sox
    # speed` and nothing at all. One clock is one lane's word, and a plain lane
    # has passed for a clock here before (see pilot_bin), so a single clock
    # buys the cheap mistake rather than the dear one and says so.
    said = [m for *_, m in got]
    measured = said[0] if len(said) == 2 and said[0] == said[1] else 1
    if rows is None and measured == 1 and max(said) > 1:
        print(f"a clock says {max(said)} rows to the sample and that takes two "
              f"agreeing; read as 1, which is the mistake that undoes itself. "
              f"If the sheet is that, say --rows {max(said)}")
    rows = measured if rows is None else rows
    return thin(song, sr, rows), n - len(js), js, turned, rows


def lane_joins(songs):
    """Every sample index where one lane meets the next, sheets included.

    A lane is not a second on every sheet -- `--rows 2` puts two lanes in one --
    so the joins are counted per sheet rather than assumed `rate` apart.

    The seam BETWEEN two sheets is a join like any other, and is the one this
    used to miss: a sheet of n lanes ends at n*lane, which is where the walk up
    that sheet stops rather than a step it takes. It is also the join most
    likely to be heard -- a sheet is around two minutes of audio, so it was one
    click every two minutes, on exactly the reads that are long enough to sit
    through. The last sheet's end is not a join at all, being where the file
    stops.
    """
    joins, at = [], 0
    for piece, lane in songs:
        joins += range(at + lane, at + len(piece), lane)
        at += len(piece)
        joins.append(at)
    return np.array(joins[:-1], int)


def do_read(args):
    songs, rate = [], None
    for path in args.scans:
        ink = load_ink(path)
        if args.upside_down:
            # The sheet went on the glass the wrong way round, so the scan is
            # the page turned 180 degrees: time runs up the lanes and the lanes
            # run right to left. Undoing it is that same turn, applied to the
            # array rather than to the file -- the scan on disk stays original.
            ink = ink[::-1, ::-1]
        note = retouched(ink)
        if note:
            print(f"{path}: {note}")
        raw, sr, n0 = read_curves_page(ink, args.pitch, args.sr,
                                       narrow=args.aperture)
        song, n, js, turned, nrows = pilot_retime(raw, sr, n0, args.pilot,
                                                  args.rows)
        if turned:
            print(f"{path}: the clocks read back in the wrong order -- the "
                  f"sheet went on the glass upside down; the read has been "
                  f"turned. The file on disk is untouched")
        if args.pilot and not js:
            # Not fatal: a sheet printed with --no-pilot has none and reads
            # fine without them. What it is worth saying for is the other case,
            # where the sheet does carry clocks and the grid was miscounted
            # badly enough that the end lanes are not them -- the read comes
            # back the right length, sounds like the recording, and every lane
            # is a fraction of a lane out.
            print(f"{path}: no clock lanes found -- printed without them, or "
                  f"the lane grid is miscounted. Reading untimed")
        if js:
            j = np.mean(js, axis=0)
            # rms, not peak to peak: the estimate has a thin tail of outliers
            # and a peak figure is all tail. On a sheet that never left the
            # computer this reads 0.08 samples rms against 1.47 on a real scan.
            print(f"{path}: {len(js)} clock lane{'s' if len(js) != 1 else ''} found, "
                  f"carriage ripple {j.std():.2f} samples rms")
            if len(js) == 1:
                # A sheet has two, so look a few lanes in from each end before
                # blaming damage. A clock short of the edge is the segmentation
                # having overrun the field -- the scanner's own edge, cut at
                # the sheet's own pitch -- and every lane past it is a second of
                # nothing on the end of the wav. Read off `raw`, which is the
                # sheet's own frame here: a turn needs both clocks, so a read
                # with one of them was never turned.
                edge = 6
                look = (*range(min(edge, n0)), *range(max(edge, n0 - edge), n0))
                at = [k for k in look
                      if pilot_bin(raw[k * sr:(k + 1) * sr]) is not None]
                if len(at) > 1:
                    print(f"{path}: clocks on lanes {at[0]} and {at[-1]} of "
                          f"{n0} -- the lane grid overran "
                          f"the field by {at[0]} on the left and "
                          f"{n0 - 1 - at[-1]} on the right. Those lanes are "
                          f"seconds of nothing in this read, and only the one "
                          f"clock was used")
                else:
                    print(f"{path}: only one clock -- a sheet has two, so "
                          f"either the other lane is damaged or the lanes were "
                          f"miscounted")
        print(f"{path}: {ink.shape[1]}x{ink.shape[0]} px, {n} lanes, {sr} Hz"
              + (f", {nrows} rows a sample = {1 / nrows:g} s a lane"
                 if nrows > 1 else ""))
        lane = sr // nrows           # samples a lane holds, for the declick
        # A lane holds `lane` samples and lasts 1/nrows of a second, so this is
        # the rate whatever the scan's ink height came back as. At one row a
        # sample it is sr unchanged; at two it drops the odd row rather than
        # letting the read run half a sample short every lane.
        sr = lane * nrows
        if rate is None:
            rate = sr
        elif sr != rate:
            # Sheet two may have gone through the scanner a percent larger, and
            # a percent of rate is a percent of pitch: one lane is one second on
            # every sheet, so stretching each sheet to the first sheet's rate is
            # exact, not a guess.
            print(f"  stretched to the first sheet's {rate} Hz")
            song = np.interp(np.linspace(0, len(song) - 1, int(round(len(song) * rate / sr))),
                             np.arange(len(song)), song)
            lane = round(lane * rate / sr)
        songs.append((song, lane))

    song = np.concatenate([piece for piece, _ in songs])
    joins = lane_joins(songs)
    song = declick(highpass(despeckle(song, args.despeckle), BASELINE),
                   joins, DECLICK)
    out = args.out or args.scans[0].rsplit(".", 1)[0] + ".wav"
    write_wav(out, song, rate)
    print(f"wrote {out}: {len(song)} samples, {len(song) / rate:.1f} s at {rate} Hz")


# -------------------------------------------------------------------- selftest

def selftest():
    """A sheet through the whole path: audio -> PNG -> audio, files and all."""
    import contextlib
    import io as _io
    import tempfile

    # A 38.1 mm square at 600 dpi is 25.4 mm of printable paper: 600 rows, so
    # 600 Hz, and fifteen lanes of the default pitch across it.
    width, rate = sheet_px("38.1x38.1", 600, 6.35)
    assert (width, rate) == (600, 600), f"printable area {width}x{rate}"
    # The sheet's margin is wider than a pitch, so it is the blank the outer
    # lane needs and nothing extra is drawn for it.
    edge = max(0.0, PITCH - mm_px(6.35, 600))
    assert edge == 0, "a 6.35 mm margin should already be blank enough"
    lanes = lanes_that_fit(width, PITCH, edge)
    assert lanes == 15, f"{lanes} lanes fit, want 15"

    rng = np.random.default_rng(5)
    sig = box1d(rng.normal(size=(1, lanes * rate)), 9, 1)[0]
    sig = 0.7 * sig / np.abs(sig).max()

    # Resampling has to filter before it decimates: a tone above the new
    # Nyquist must come back quiet, not folded down into the audio as a tone
    # that was never played.
    t = np.arange(4 * 4800) / 4800
    high = to_rate(np.sin(2 * np.pi * 2000 * t), 4800, 600)
    assert np.abs(high).max() < 0.3, (
        f"2 kHz folded into a 600 Hz sheet at {np.abs(high).max():.2f}")

    with tempfile.TemporaryDirectory() as tmp:
        wav, png = os.path.join(tmp, "in.wav"), os.path.join(tmp, "sheet.png")
        png2 = os.path.join(tmp, "sheet2.png")
        write_wav(wav, sig, rate)
        args = argparse.Namespace(audio=wav, out=png, paper="38.1x38.1", dpi=600,
                                  margin_mm=6.35, pitch=PITCH, start=0.0,
                                  pilot=False)
        with contextlib.redirect_stdout(_io.StringIO()):
            do_print(args)
            args.out, args.start = png2, 3.0     # the next sheet of a long one
            do_print(args)
        ink = load_ink(png)
        # The file is the whole sheet, margins and all, and the ink inside it
        # is exactly as tall as the rate: 38.1 mm of paper at 600 dpi carrying
        # 25.4 mm of lanes.
        sheet = mm_px(38.1, 600)
        assert ink.shape == (sheet, sheet), f"sheet is {ink.shape}, want {sheet} square"
        rows = np.flatnonzero(ink.sum(1) > 0)
        assert rows[-1] + 1 - rows[0] == rate, (
            f"ink is {rows[-1] + 1 - rows[0]} rows, want {rate}")
        assert abs(rows[0] - (sheet - rate) / 2) <= 1, "ink is not centred on the sheet"
        # both sheets at the same size, or the sound changes pitch at the join
        assert load_ink(png2).shape == ink.shape, "sheets disagree on size"

        # And the seam BETWEEN two sheets is a lane join like any other -- the
        # one that used to be missed, because it is where the walk up a sheet
        # stops rather than a step it takes. Two sheets of three and two lanes
        # join at every lane and at the seam, and not at the end of the file.
        two = [(np.zeros(3 * rate), rate), (np.zeros(2 * rate), rate)]
        assert list(lane_joins(two)) == [rate, 2 * rate, 3 * rate, 4 * rate], (
            f"lane joins over two sheets: {list(lane_joins(two))}")

        # A read that is mostly silence has no median step to threshold
        # against, and a threshold of zero would flatten what is left of it.
        quiet = np.zeros(200)
        quiet[100] = 1.0
        assert despeckle(quiet, 20)[100] == 1.0, (
            "despeckle on a silent read took out the only sample in it")

        # A blockless sheet of a single lane cannot be segmented -- the pitch
        # estimator searches from two periods up -- so printing pads it out to
        # two rather than handing over a sheet nothing can read.
        write_wav(wav, sig[:rate], rate)     # one second: a one-lane print
        args.out, args.start = png2, 0.0
        said = _io.StringIO()
        with contextlib.redirect_stdout(said):
            do_print(args)
        assert "2 lanes" in said.getvalue(), (
            f"a lone blockless lane was printed: {said.getvalue().strip()}")

    got, out_rate, n = read_curves_page(ink)
    assert (out_rate, n) == (rate, lanes), f"read {n} lanes at {out_rate} Hz"

    # Print and scan fade the first and last rows of ink into the paper around
    # it. Those rows are samples, and cutting them off as margin loses one from
    # every lane and 0.02 of the read, so the trim has to keep them.
    faded = box1d(box1d(ink.astype(np.float64), 3, 0), 3, 1)
    assert read_curves_page(faded)[1] == rate, (
        f"faded ink edge read as {read_curves_page(faded)[1]} Hz, want {rate}")
    r = np.corrcoef(highpass(got, 31), highpass(sig[:len(got)], 31))[0, 1]
    assert r > 0.95, f"round trip correlation {r:.4f}"

    # The clock lane, against a sheet whose rows have been rippled the way a
    # scanner's carriage ripples them. This is the one defect on the bench that
    # nothing else here can see: it leaves no horizontal trace at all.
    with tempfile.TemporaryDirectory() as tmp:
        wav, png = os.path.join(tmp, "in.wav"), os.path.join(tmp, "clock.png")
        write_wav(wav, sig[:8 * rate], rate)
        args = argparse.Namespace(audio=wav, out=png, paper="38.1x38.1", dpi=600,
                                  margin_mm=6.35, pitch=PITCH, start=0.0,
                                  pilot=True)
        with contextlib.redirect_stdout(_io.StringIO()):
            do_print(args)
        page = load_ink(png).astype(np.float64)

    h = len(page)
    ripple = 2.0 * np.sin(2 * np.pi * 3 * np.arange(h) / h)   # rows, as measured
    i = np.clip(np.floor(np.arange(h) + ripple), 0, h - 2).astype(int)
    f = (np.arange(h) + ripple - i)[:, None]
    warped = page[i] * (1 - f) + page[i + 1] * f

    got, out_rate, n = read_curves_page(warped)
    assert n == 10, f"clock sheet read {n} lanes of 10"     # 8 seconds, 2 clocks
    ends = [pilot_timebase(got[k * out_rate:(k + 1) * out_rate]) for k in (0, n - 1)]
    assert all(e is not None for e in ends), "a clock lane was not recognised"
    # The rates are the orientation mark, so each clock has to come back as
    # itself, not merely as a clock.
    assert [per for _, per, _ in ends] == [PILOT, PILOT_END], (
        f"clock rates read back as {[per for _, per, _ in ends]}")
    assert [m for _, _, m in ends] == [1, 1], "a one-row sheet read back slowed"
    j = np.mean([j for j, _, _ in ends], axis=0)
    assert abs((j.max() - j.min()) - 4.0) < 1.0, (
        f"ripple read as {j.max() - j.min():.2f} samples p2p, drew 4.0")
    want = sig[:8 * out_rate]
    before = np.corrcoef(highpass(got[out_rate:-out_rate], 31),
                         highpass(want, 31))[0, 1]
    after = retime(got, out_rate, j)[out_rate:-out_rate]
    after = np.corrcoef(highpass(after, 31), highpass(want, 31))[0, 1]
    assert after > before + 0.05, f"clock bought nothing: {before:.3f} -> {after:.3f}"
    assert after > 0.95, f"retimed sheet still only {after:.4f}"
    assert pilot_timebase(got[out_rate:2 * out_rate]) is None, "audio read as a clock"

    # The sheet upside down on the glass: the clocks read back in the wrong
    # order, and pilot_retime turns the read itself -- reversing the signal
    # and its sign is the 180 degree turn.
    assert not pilot_retime(got, out_rate, n)[3], "a right-way-up sheet was turned"
    got_u, rate_u, n_u = read_curves_page(page[::-1, ::-1])
    fixed, n_u, js_u, turned, _ = pilot_retime(got_u, rate_u, n_u)
    assert turned, "an upside-down sheet went unnoticed"
    assert len(js_u) == 2 and n_u == 8, f"turned read kept {n_u} lanes, {len(js_u)} clocks"
    ru = np.corrcoef(highpass(fixed, 31), highpass(want, 31))[0, 1]
    assert ru > 0.95, f"turned read scores {ru:.4f}"
    # One clock found, and not at the head -- a damaged left clock, or a grid
    # that overran on the left. The end that carried it is the end to cut.
    lame = got.copy()
    lame[:out_rate] = got[out_rate:2 * out_rate]   # head clock scribbled over
    cut, n_l, js_l, _, _ = pilot_retime(lame, out_rate, n)
    assert len(js_l) == 1 and n_l == n - 1, f"{len(js_l)} clocks, {n_l} lanes"
    assert len(cut) == len(lame) - out_rate, "one clock cut two lanes"
    assert pilot_bin(cut[-out_rate:]) is None, "the head was cut and the clock kept"
    # The trace of a real lane is a position, and its slow wander towers over
    # everything else in it: on a real scan the peak bin of every music lane
    # is bin one, so an energy test alone calls every one of them a clock.
    wander = np.cumsum(rng.normal(size=out_rate))
    assert pilot_bin(wander / np.abs(wander).max()) is None, "wander read as a clock"

    # Two rows to the sample: the sheet holds half the seconds, says so on its
    # own clocks and nowhere else, and comes back as the same audio at the same
    # rate -- half the samples, playing for half as long.
    with tempfile.TemporaryDirectory() as tmp:
        wav, png = os.path.join(tmp, "in.wav"), os.path.join(tmp, "slow.png")
        write_wav(wav, sig[:4 * rate], rate)
        args = argparse.Namespace(audio=wav, out=png, paper="38.1x38.1", dpi=600,
                                  margin_mm=6.35, pitch=PITCH, start=0.0,
                                  pilot=True, rows=2)
        with contextlib.redirect_stdout(_io.StringIO()):
            do_print(args)
        slow, slow_rate, n2 = read_curves_page(load_ink(png))
        down = read_curves_page(load_ink(png)[::-1, ::-1])
    assert n2 == 10, f"4 s at two rows a sample cut into {n2} lanes, want 10"
    got2, n2, js2, _, got_rows = pilot_retime(slow, slow_rate, n2)
    assert got_rows == 2, f"a two-row sheet read back as {got_rows} rows a sample"
    assert len(got2) == n2 * (slow_rate // 2), (
        f"{len(got2)} samples in {n2} lanes of {slow_rate // 2}")
    r2 = np.corrcoef(highpass(got2, 31), highpass(sig[:len(got2)], 31))[0, 1]
    assert r2 > 0.95, f"two rows a sample round trip {r2:.4f}"
    # And the same sheet face down on the glass: the turn and the speed come
    # off the same two clocks, so a sheet that loses one loses both.
    got3, _, _, turned3, rows3 = pilot_retime(*down)
    assert turned3 and rows3 == 2, (
        f"turned two-row sheet read as turned={turned3}, {rows3} rows a sample")
    r3 = np.corrcoef(highpass(got3, 31), highpass(sig[:len(got3)], 31))[0, 1]
    assert r3 > 0.95, f"turned two-row round trip {r3:.4f}"

    # Two clocks at the same rate: the sheet is symmetric after all, the reader
    # cannot know, and the one thing it must not do is decide in silence.
    same = got.copy()
    same[-out_rate:] = got[:out_rate]
    said = _io.StringIO()
    with contextlib.redirect_stdout(said):
        assert not pilot_retime(same, out_rate, n)[3], "a symmetric sheet was turned"
    assert "--upside-down" in said.getvalue(), (
        f"a sheet that cannot say which way up it lay said nothing: "
        f"{said.getvalue().strip()!r}")

    # One clock left, saying two rows a sample. That is one lane's word, and
    # the mistake it could buy is the expensive one, so the read comes back at
    # full height with the flag to fix it named -- and the flag works.
    lame2 = slow.copy()
    lame2[:slow_rate] = slow[slow_rate:2 * slow_rate]   # head clock scribbled on
    said = _io.StringIO()
    with contextlib.redirect_stdout(said):
        kept, _, js4, _, rows4 = pilot_retime(lame2, slow_rate, n2)
    assert (rows4, len(js4)) == (1, 1), f"one clock thinned on its own word"
    assert "--rows 2" in said.getvalue(), (
        f"a lone clock was overruled in silence: {said.getvalue().strip()!r}")
    forced = pilot_retime(lame2, slow_rate, n2, rows=2)[0]
    assert len(forced) == len(kept) // 2, (
        f"--rows 2 gave {len(forced)} samples against {len(kept) // 2}")
    # And with no clocks at all -- the sheet that cannot say anything -- the
    # flag is the only thing left that can.
    bare = pilot_retime(slow, slow_rate, n2, pilot=False, rows=2)[0]
    assert len(bare) == len(slow) // slow_rate * (slow_rate // 2), (
        f"a clockless read forced to two rows came back {len(bare)} long")

    # A sheet rendered with a skew drawn into it, which is now the reader's own
    # problem: nothing on the page says how crooked it is. What finding it is
    # worth depends on the sheet -- 2.34 dB on the crooked sheet at 17.7 px of
    # skew, 0.09 dB on the rippled sheet at 4.9 px -- so this checks the
    # mechanism at a skew that makes it matter rather than at a typical one.
    #
    # The last of these is the check on it. 24 px of drift down a 600-row sheet
    # reads back at 0.9998 corrected and 0.8737 with the estimate forced to
    # zero, so the assert below fails if it ever stops working. The two smaller
    # skews do not test it at all -- the sticky window tracks 12 px of drift on
    # its own and still returns 0.963 uncorrected -- which is why the value
    # here is 0.04 and not the 0.0025 that a crooked sheet actually looks like.
    #
    # The drift is aimed off the ink's own edges and then fitted per strip of
    # rows -- lane_drift() carries what that is worth. What is left over for
    # these three sheets is that the mechanism works at a skew that matters:
    # 24 px down a 600-row sheet reads 0.9998 corrected against 0.8737 with
    # the estimate forced to zero. The smaller two do not test it at all, the
    # sticky window tracks 12 px on its own, which is why the value here is
    # 0.04 and not the 0.0025 a crooked sheet actually looks like.
    #
    # The ceiling past these used to be the lane pitch, and it belonged to the
    # cut rather than to the fit: at 40 px of shear on a 38.7 px pitch the page
    # cut into twelve lanes where eleven were printed. It is gone -- the
    # profile the grid is cut on is summed along the lean now -- and the sheets
    # below it check that, at 48 and at 300 px.
    step0, nlanes = 39.0, 11
    for drawn in (0.0, 0.0025, 0.04):    # 0, 1.5 and 24 px down the sheet
        clocked = np.concatenate([pilot_lane(rate), sig[:(nlanes - 2) * rate],
                                  pilot_lane(rate)])
        edges, amp = lay_out(clocked, rate, nlanes, step0, drawn)
        wide = int(np.ceil(max(r.max() for _, r in edges) + step0))
        sheet_ink = trim_paper(render_page(edges, step0, wide) * 255)

        assert len(find_tracks(sheet_ink)) == nlanes, (
            f"skew {drawn:+.4f}: sheet cut into {len(find_tracks(sheet_ink))} lanes")
        said = _io.StringIO()
        with contextlib.redirect_stdout(said):
            got, _, got_n = read_curves_page(sheet_ink)
        # The fit holds on all three, and says so by keeping quiet.
        assert "Rescan it straighter" not in said.getvalue(), (
            f"skew {drawn:+.4f} reads back at r>0.95 and was called crooked: "
            f"{said.getvalue().strip()!r}")
        assert got_n == nlanes, f"skewed sheet read {got_n} lanes of {nlanes}"
        rr = np.corrcoef(highpass(got[rate:-rate], 31),
                         highpass(sig[:(nlanes - 2) * rate], 31))[0, 1]
        assert rr > 0.95, f"skew {drawn:+.5f}: read back at only {rr:.4f}"

    # And past where the cut used to give up: 48 px of shear on a 39 px pitch,
    # a lane at one end of the page over its neighbour's columns at the other,
    # and 300 px, which is 27 degrees and further than a sheet of paper can be
    # laid on a platen. Both read back whole. The column profile is summed
    # along the ink's own lean instead of straight down (read_tracks.
    # col_profile), so the periodicity the grid is cut on survives a shear the
    # page cannot physically carry -- no pixel is resampled to do it, which is
    # what the 8.6 dB white point says about touching the ink.
    #
    # This used to be the check that a hopeless sheet SAID so, and the warning
    # it checked for is gone with the ceiling it named. What is left to say
    # about skew is the fit leaving the sheet (SKEW_GAP, above), which is a
    # broken measurement rather than a sheet past the format.
    for shear in (0.08, 0.5):
        clocked = np.concatenate([pilot_lane(rate), sig[:(nlanes - 2) * rate],
                                  pilot_lane(rate)])
        edges, _ = lay_out(clocked, rate, nlanes, step0, shear)
        wide = int(np.ceil(max(r.max() for _, r in edges) + step0))
        bent = trim_paper(render_page(edges, step0, wide) * 255)
        said = _io.StringIO()
        with contextlib.redirect_stdout(said):
            got4, _, got_n = read_curves_page(bent)
        assert got_n == nlanes, (
            f"{shear * rate:.0f} px of shear cut {got_n} lanes of {nlanes}")
        m = min(len(got4) - 2 * rate, (nlanes - 2) * rate)
        rr = np.corrcoef(highpass(got4[rate:rate + m], 31),
                         highpass(sig[:m], 31))[0, 1]
        assert rr > 0.95, (
            f"{shear * rate:.0f} px of shear read back at only {rr:.4f}")
        assert "Rescan" not in said.getvalue(), (
            f"{shear * rate:.0f} px of shear, read whole and called crooked: "
            f"{said.getvalue().strip()!r}")

    # A PNG whose ancillary chunk carries a CRC its writer got wrong. One
    # scanner here does that to pHYs -- the chunk that says dpi and nothing
    # else, which nothing in this reader reads -- and PIL refuses the whole
    # image over it. Mended in memory; a critical chunk is not ours to mend.
    from PIL import Image as _Image
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.png")
        _Image.fromarray(np.uint8(sheet_ink[:200, :200])).save(bad, dpi=(600, 600))
        raw = bytearray(open(bad, "rb").read())

        def chunk_at(name):
            i = 8
            while i + 12 <= len(raw):
                ln = int.from_bytes(raw[i:i + 4], "big")
                if bytes(raw[i + 4:i + 8]) == name:
                    return i + 8 + ln
                i += 12 + ln
            raise AssertionError(f"{name} not in the test PNG")

        # A 16-bit greyscale scan, which is what a flatbed set to its best
        # depth writes and what PIL's convert("L") turns into a blank sheet by
        # clipping instead of scaling. The same page at both depths has to read
        # back as the same page.
        deep = os.path.join(tmp, "deep.png")
        page8 = np.uint8(sheet_ink[:200, :200])
        _Image.fromarray(page8.astype(np.uint16) * 257).save(deep)
        assert np.abs(load_ink(deep).astype(int)
                      - load_ink(bad).astype(int)).max() <= 1, (
            "a 16-bit scan did not read back as the 8-bit page it is")

        end = chunk_at(b"pHYs")
        raw[end:end + 4] = bytes(4)          # the CRC, wrong on purpose
        open(bad, "wb").write(bytes(raw))
        said = _io.StringIO()
        with contextlib.redirect_stdout(said):
            back = load_ink(bad)
        assert back.shape == (200, 200), f"a mended PNG read back as {back.shape}"
        assert "CRC" in said.getvalue(), (
            f"a chunk was mended in silence: {said.getvalue().strip()!r}")
        end = chunk_at(b"IDAT")
        raw[end:end + 4] = bytes(4)          # critical: must NOT be mended
        open(bad, "wb").write(bytes(raw))
        try:
            load_ink(bad)
        except Exception:
            pass
        else:
            raise AssertionError("a broken IDAT was read as if it were sound")

    # A lane wiped off the sheet. Silence has to go in its place: dropped
    # instead, the read comes back a second short and everything past the hole
    # plays a second early -- which still sounds like the recording and scores
    # as noise from the hole on. The numbers are on the fill itself; this is
    # what keeps them true, because the failure is invisible in the total.
    clocked = np.concatenate([pilot_lane(rate), sig[:(nlanes - 2) * rate],
                              pilot_lane(rate, PILOT_END)])
    edges, _ = lay_out(clocked, rate, nlanes, step0, 0.0)
    wide = int(np.ceil(max(r.max() for _, r in edges) + step0))
    holed = trim_paper(render_page(edges, step0, wide) * 255)
    x0, x1 = find_tracks(holed)[5]        # lane 5 of the page is second 4
    holed[:, x0:x1] = 0
    said = _io.StringIO()
    with contextlib.redirect_stdout(said):
        got_h, rate_h, n_h = read_curves_page(holed)
    assert n_h == nlanes, f"a wiped lane left {n_h} lanes of {nlanes}"
    assert "no lane on the sheet to read there" in said.getvalue(), (
        f"the hole was filled without a word: {said.getvalue().strip()!r}")
    # The lane after the hole has to be its own second. Were the hole dropped
    # instead of filled, this is where the NEXT second would have landed, so
    # the two correlations tell the two outcomes apart on their own.
    past = highpass(got_h[6 * rate_h:7 * rate_h], 31)
    kept = np.corrcoef(past, highpass(sig[5 * rate:6 * rate], 31))[0, 1]
    slid = np.corrcoef(past, highpass(sig[6 * rate:7 * rate], 31))[0, 1]
    assert kept > 0.95 > slid, (
        f"past the hole the read sits at {kept:.3f} against its own second and "
        f"{slid:.3f} against the one after it -- the tail slid by a lane")

    # A scan that has been through a levels tool. A rendered sheet uses every
    # value between paper and ink; stretching it spreads the same values over a
    # wider range and leaves a comb of empty bins, which is the whole of what
    # retouched() can see.
    assert retouched(ink) is None, "a rendered sheet was called retouched"
    stretched = np.minimum(ink.astype(np.int32) * 3, 255).astype(np.uint8)
    assert retouched(stretched), "a 3x levels stretch was not spotted"

    # A scan whose edges have been squared off, which retouched() cannot see:
    # the curve below is applied before quantising, so it leaves no comb. What
    # gives it away is the stroke, which comes back narrower than it was drawn
    # -- and ink on paper has never done that.
    grid = find_tracks(ink)
    assert steepened(ink, grid) is None, "a rendered sheet was called steepened"
    assert abs(stroke_fwhm(ink, grid) - STROKE) < 0.05, "drawn width moved"
    hard = (255 * (ink / 255.0) ** 2).astype(np.uint8)
    assert steepened(hard, grid), "squared-off edges were not spotted"
    wide = (255 * (ink / 255.0) ** 0.5).astype(np.uint8)
    assert steepened(wide, grid) is None, "a WIDENED stroke is not this defect"
    # A straight stretch must not register, and this is the property that makes
    # the check worth having: brightness and contrast do not move a half
    # maximum, only CURVATURE does. 3x - 1.0 leaves the stroke on 4.00 px while
    # leaving retouched()'s comb behind, so the two checks see different things
    # by construction rather than by luck.
    flat = (255 * np.clip(ink / 255.0 * 3 - 1.0, 0, 1)).astype(np.uint8)
    assert steepened(flat, grid) is None, "a linear stretch was called steep"
    assert retouched(flat), "the linear stretch should still leave a comb"

    # Something on the sheet that is not a lane, segmented as if it were: the
    # failure a mark printed beside the field causes. A mark six pixels wide
    # beside the field comes back as a lane of its own, and every second after
    # it is shifted by one.
    #
    # The mark is drawn at two thirds of full ink on purpose, so that six
    # columns of it weigh what four columns of stroke weigh. find_tracks bounds
    # a lane's ink from above as well as below now, and a mark laid at full
    # strength is simply thrown out there -- which is the better answer, and
    # leaves this test measuring nothing. What is left for the width warning is
    # exactly this: a mark carrying a lane's worth of ink in the wrong shape.
    marked = np.zeros((len(sheet_ink), sheet_ink.shape[1] + 20), sheet_ink.dtype)
    marked[:, 6:12] = 2 * sheet_ink.max() // 3
    marked[:, 20:] = sheet_ink
    said = _io.StringIO()
    with contextlib.redirect_stdout(said):
        read_curves_page(marked)
    assert "segmented as a lane that is not one" in said.getvalue(), (
        f"a 6 px mark was read as a lane in silence: {said.getvalue().strip()!r}")

    # And the one the width warning cannot catch, because it is a lane wide and
    # at the sheet's own pitch: the scanner's own edge past the end of the
    # field. It is thrown out by weight -- a lane's ink is a stroke, not a
    # slab. Left in, it is a second of nothing in the read, and on a real sheet
    # here it also hid the right-hand clock and cost the retiming one end.
    edged = np.zeros((len(sheet_ink), sheet_ink.shape[1] + 60), sheet_ink.dtype)
    edged[:, :sheet_ink.shape[1]] = sheet_ink
    edged[:, -30:] = sheet_ink.max()
    assert len(find_tracks(edged)) == nlanes, (
        f"a solid edge added {len(find_tracks(edged)) - nlanes} lanes")

    # Impulse rejection: a blot's worth of samples out of the trace and back,
    # flattened, and the signal either side of it left exactly where it was.
    clean = np.sin(2 * np.pi * 7 * np.arange(600) / 600)
    dirty = clean.copy()
    dirty[300:306] += 5.0
    fixed = despeckle(dirty, 20)
    assert np.abs(dirty - clean).max() == 5.0, "the bench blot is not 5 px"
    # What is left is the two replacement steps, not the blot: the trace either
    # side of it keeps its level to a fraction of a sample.
    assert np.abs(fixed - clean).max() < 0.2, "despeckle left the blot in"
    assert np.array_equal(despeckle(dirty, 0), dirty), "despeckle ran when off"

    print(f"selftest ok: {n} lanes at {out_rate} Hz, round trip {r:.4f}, "
          f"clock {before:.3f} -> {after:.3f}, two rows a sample {r2:.4f}")


# ------------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("print", help="audio -> a sheet to print")
    pr.add_argument("audio", help="16-bit PCM wav")
    pr.add_argument("-o", "--out", help="output PNG (default: <audio>.png)")
    pr.add_argument("--paper", default=PAPER_DEFAULT,
                    help=f"{', '.join(PAPER)}, or WxH in mm (default {PAPER_DEFAULT})")
    pr.add_argument("--dpi", type=int, default=DPI,
                    help=f"print resolution (default {DPI})")
    pr.add_argument("--margin-mm", type=float, default=MARGIN_MM,
                    help=f"mm the printer cannot reach (default {MARGIN_MM:g})")
    pr.add_argument("--pitch", type=float, default=PITCH,
                    help=f"lane pitch in px (default {PITCH:g} = "
                         f"{PITCH * 25.4 / DPI:.2f} mm at {DPI} dpi)")
    pr.add_argument("--rows", type=int, default=1, choices=ROWS,
                    help="printed rows per audio sample (default: 1). 2 halves "
                         "the ink's slope per row and the seconds a sheet "
                         "holds; the clocks carry it, so `read` needs no flag")
    pr.add_argument("--start", type=float, default=0.0,
                    help="seconds to skip before the first sheet")

    rd = sub.add_parser("read", help="a scan -> audio")
    rd.add_argument("scans", nargs="+", help="scanned sheets, in playing order")
    rd.add_argument("-o", "--out", help="output WAV (default: <first scan>.wav)")
    rd.add_argument("--pitch", type=float, help="lane pitch in px (default: measured)")
    rd.add_argument("--upside-down", action="store_true",
                    help="the sheet was scanned the wrong way round: turn every "
                         "scan 180 degrees before reading it. Sheets whose two "
                         "clocks differ turn themselves; this is for one printed "
                         "without clocks, or before the clocks differed")
    rd.add_argument("--rows", type=int, choices=ROWS,
                    help="rows per audio sample, when the clocks cannot say "
                         "(default: measured off them). A sheet printed with "
                         "--rows 2 whose clocks are damaged needs this: "
                         "nothing else on it carries the number")
    rd.add_argument("--sr", type=int, help="samples per lane (default: measured)")
    rd.add_argument("--aperture", type=float, default=None,
                    help=f"px of lane the centroid looks at; narrower rejects "
                         f"grime, wider sees all of the stroke. Default is "
                         f"{APERTURE:g} px scaled by the sheet's own pitch, "
                         f"which is what the three geometries measured here "
                         f"want; a number given here is taken as it stands")
    rd.add_argument("--no-pilot", dest="pilot", action="store_false", default=True,
                    help="ignore the sheet's clock lane and read it as audio")
    rd.add_argument("--despeckle", type=float, default=DESPECKLE,
                    help=f"multiple of the median step a sample may jump before "
                         f"it is read as dust, 0 off (default {DESPECKLE:g}). "
                         f"For a genuinely filthy page; on a clean one it only "
                         f"takes signal")

    sub.add_parser("selftest", help="print a sheet, read it back, check it")

    args = p.parse_args()
    try:
        if args.cmd == "selftest":
            selftest()
        else:
            (do_print if args.cmd == "print" else do_read)(args)
    except ValueError as e:
        raise SystemExit(f"{args.cmd}: {e}")     # an unreadable sheet is an
                                                 # answer, not a stack trace


if __name__ == "__main__":
    main()
