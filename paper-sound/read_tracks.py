#!/usr/bin/env python3
"""Raster work: getting a scanned page off disk and cut into lanes.

Everything here is about pixels rather than about sound. paper_sound.py holds
the format and calls into this; nothing here knows what a lane means.

Two jobs. Finding the lanes, which is done by measuring the page's own
periodicity rather than by looking for gaps, because on a sheet of curves there
need not be a gap -- two neighbouring strokes can both swing towards each other
until the white between them closes. And the small numeric helpers a reader
needs afterwards: resampling a lane to a common length, subtracting a baseline,
and easing the joins where one lane's second meets the next.

Sub-pixel throughout. The signal on a printed sheet is a few pixels of stroke
position, so rounding anything to a whole pixel here would throw away most of
it.
"""
import wave

import io
import zlib

import numpy as np
from PIL import Image, UnidentifiedImageError

INK = 0.02    # ink threshold, as a fraction of the strongest ink present
HEADROOM = 1.3  # how far above the 99.9th percentile write_wav() puts full
                # scale. Speech reaches 1.2 of its own 99.9th percentile and a
                # read that lost lanes reaches 3.6; 1.3 clips nothing in the
                # first case and only the lost lanes in the second. See
                # write_wav().
SUBHARM = 0.5 # how much of the winning bin a half or a third of it has to
              # carry before it is believed to be the real fundamental. See
              # find_pitch(); measured, the two populations sit either side of
              # this by a factor of 2.6. It is not a clean line -- a page whose
              # peak IS the fundamental has been seen at 0.245 and one whose
              # peak is a harmonic at 0.235 -- so this test alone cannot settle
              # it, and harmonic() asks the paper afterwards.
VETO = 0.03   # how far the median of thirds may sit from the whole page's own
              # number and still be preferred. Measured on the four scans of
              # 26.08 the thirds of a sheet differ by 0.6%; a third that has
              # landed on a harmonic of its own window differs by 40% or more,
              # and nothing here falls between.
CUTS = 1.3    # how much more ink than the cleanest candidate a pitch may put
              # on its cut lines and still be taken, for being the smaller.
              # See harmonic().
PEAK = 99     # percentile of the ROW maxima taken as the level of the ink, in
              # place of the darkest pixel anywhere. A defect has to spoil one
              # line in a hundred before it moves a threshold: 26 rows of the
              # scanner's own edge on a 6600-row page, or ten rows of speck on
              # a 1000-row lane, all of which is under it. Swept on the sheets
              # here, in dB against the read they gave off max():
              #
              #                 99.9    99     95
              #   scan_rows1   +0.94  +6.55  +6.59   (23 bad seconds -> 9)
              #   sheetC       -0.13  +0.19  +0.23
              #   sheetD_scan3 +0.02  +0.04  -2.22
              #   sheetB       -0.01  -0.77  -0.72
              #   the other 10  0.00   0.00  +0.15 and under
              #
              # 99.9 is max() again and does nothing; 95 reaches far enough
              # into the flank of the stroke to cost sheetD_scan3 its read.
              # See ink_level().


def ink_level(a, floor=0.0, q=PEAK):
    """Halfway between the paper and the ink, from the rows and not one pixel.

    Taken off max() instead -- which is what every threshold in this file used
    to do -- the level of the ink is set by ONE pixel, and any pixel darker
    than the print decides it for the whole page. A speck of dust, a hair, a
    black fleck at the edge of the glass: one of those at 255 over a stroke
    printed at 100 puts the threshold above the stroke and the ink vanishes.
    Measured on a synthetic lane, ink_rows() answered (0, 1000) clean and None
    with a single 255 pixel added to it, and ink_lean() went from a lean to no
    reading at all the same way.

    It only bites where the print is faint: a stroke at 240 has no room above
    it for a defect to hide in, and adding one changes nothing. But a faint
    print is exactly the scan that needs the reader to hold.

    What replaces the maximum is a percentile of the ROW maxima, not of the
    pixels. The pixels answer a different question -- most of a lane is paper,
    so a percentile of them lands on the flank of the stroke and quietly moves
    every threshold in the file down, which cost sheetD_scan3 2.2 dB and won
    scan_rows1 7.2, i.e. it was retuning rather than repairing. A row's darkest
    pixel is the ink, in every row, which is what max() was reaching for and
    got right on a clean scan; taking the 99th of those keeps that answer and
    takes away the single pixel's vote, since a defect has to spoil one row in
    a hundred before it counts.
    """
    return floor + 0.5 * (float(np.percentile(a.max(1), q)) - floor)


def mend_png(path):
    """A PNG with bad CRCs on ancillary chunks, mended in memory -> (bytes, names).

    One scanner here writes `pHYs` -- the chunk that carries the dpi and
    nothing else -- with a CRC it computed wrong, identically wrong in every
    file it makes. PIL refuses the whole image over it, and the image is fine:
    a sheet is measured in its own rows and this reader never asks a file what
    dpi it claims to be. So the CRC is recomputed and the read goes on.

    Critical chunks (uppercase first letter: IHDR, PLTE, IDAT, IEND) are left
    alone. A bad CRC there is a damaged image rather than a careless writer,
    and quietly reading a damaged image is the one thing worse than refusing.
    """
    d = bytearray(open(path, "rb").read())
    if bytes(d[1:4]) != b"PNG":
        return None, []
    i, mended = 8, []
    while i + 12 <= len(d):
        ln = int.from_bytes(d[i:i + 4], "big")
        typ = bytes(d[i + 4:i + 8])
        end = i + 8 + ln
        if end + 4 > len(d):
            return None, []
        want = zlib.crc32(bytes(d[i + 4:end])) & 0xffffffff
        if int.from_bytes(d[end:end + 4], "big") != want:
            if not typ[0] & 0x20:            # uppercase = critical
                return None, []
            d[end:end + 4] = want.to_bytes(4, "big")
            mended.append(typ.decode("latin1"))
        i = end + 4
        if typ == b"IEND":
            break
    return (bytes(d), mended) if mended else (None, [])


def ink_of(im):
    """One opened image -> ink intensity (paper 0, ink high), uint8.

    A 16-bit greyscale scan does NOT survive PIL's convert("L"): that path
    clips at 255 instead of scaling, so every pixel above 255 -- which on a
    16-bit scan is the paper, the ink and everything between -- comes back
    white and the page reads as blank. Measured on a 0..65535 gradient: the
    first pixel converts, the rest read 0. A flatbed set to 16-bit greyscale is
    an ordinary thing to hand this program, so it is scaled here instead.

    ponytail: scaled DOWN to 8 bits rather than carried in float, because 8 is
    what every measurement in this project was made at and the extra bits have
    never been shown to be worth anything. The carrier is the sub-pixel
    position in the anti-aliased edge, which 8 bits already resolves to about
    1/256 of a pixel; if a 16-bit sheet ever measures better, this is the line
    to change.

    A BITONAL page -- PIL's mode "1", what a scanner set to line art writes --
    comes out of np.asarray as `bool`, which is neither of the branches below
    and lands in the 16-bit one. There it is scaled as if True were 1 of 255,
    so the paper reads 254 and the ink 255: a page of solid ink one value
    apart, on which every threshold here finds whatever it likes. It goes
    through convert("L") with the 8-bit scans instead, which maps it to the
    0 and 255 it means. The README asks for a grey scan and means it -- a
    bitonal page has thrown away the anti-aliased edge this format reads, and
    what is left is a coarse read rather than a good one -- but a coarse read
    is the honest outcome, and a page of ink is not.
    """
    a = np.asarray(im)
    if a.dtype == np.uint8 or a.dtype == np.bool_:
        return 255 - np.asarray(im.convert("L"))
    a = a.astype(np.float64)
    return (255 - np.round(a * (255.0 / (65535.0 if a.max() > 255 else 255.0)))
            ).clip(0, 255).astype(np.uint8)


def load_ink(path):
    """Grayscale page -> ink intensity (paper 0, ink high), uint8."""
    try:
        with Image.open(path) as im:      # a sheet at a time, and there are many
            return ink_of(im)
    except UnidentifiedImageError:
        fixed, mended = mend_png(path)
        if fixed is None:
            raise
        print(f"{path}: {', '.join(mended)} carries a CRC its writer computed "
              f"wrong; mended in memory, the file on disk is untouched")
        with Image.open(io.BytesIO(fixed)) as im:
            return ink_of(im)


def inked_span(prof, frac=INK):
    """First and last index of prof carrying ink."""
    on = np.flatnonzero(prof > prof.max() * frac)
    if not len(on):
        raise ValueError("no ink on this page")
    return on[0], on[-1] + 1


def ink_lean(ink, bands=8):
    """How far the ink block itself leans, px per row, from each edge on its own.

    Geometry rather than content: a sheared page carries its whole block of
    ink over with it, so the leftmost and rightmost inked columns in the top
    band of the page sit somewhere else in the bottom band, and the difference
    is the shear. Nothing here asks what was printed, which is the point --
    every other measurement of the skew in this file is a centroid of the
    audio, and inherits the audio.

    The two edges are returned separately, never averaged. One of them can be
    something other than the sheet: `sheetB_scan` carries a blob in its top
    band that puts its left edge 1343 px out while its right edge lands within
    3 px of the truth. Whoever uses these takes the smaller one, so junk that
    invents a lean is thrown out and junk that hides one costs only the head
    start.
    """
    rows = len(ink)
    band = max(rows // bands, 1)
    strong = ink > ink_level(ink, float(np.median(ink[::16, ::16])))
    ends = []
    for lo in (0, rows - band):
        col = strong[lo:lo + band].sum(0)
        lit = np.flatnonzero(col > col.max() * 0.02)
        if not len(lit):
            return []
        ends.append((lit[0], lit[-1]))
    return [(b - a) / (rows - band) for a, b in zip(*ends)]


def col_profile(ink, lean=0.0):
    """Mean column profile, summed along `lean` px per row instead of straight down.

    Summed straight down, the profile is the one thing on the page that skew
    destroys: a lane at the bottom of a sheared sheet sits over its neighbour's
    columns at the top, the periodicity smears, and the grid cuts lanes that
    were never printed. Everything else in the reader already rides the lean --
    the window follows the drift, each lane is cropped to its own ink rows --
    so shifting each row by -lean*y before it is added is the whole of the fix.
    Index arithmetic, not a rotation: no pixel is resampled and no sub-pixel
    position is quantised, which is what the carrier is made of.

    Centred on mid-page, like everything else built off this profile: the
    columns that come back are where a lane sits halfway down the sheet, which
    is where lane_centroid's ramp is zero.
    """
    if not lean:
        return ink.mean(0).astype(np.float64)
    n = ink.shape[1]
    shift = np.round(lean * (np.arange(len(ink)) - len(ink) / 2)).astype(int)
    prof = np.zeros(n)
    # Divided by the rows that actually LANDED in each column, not by the page
    # height: a column near the edge is fed by only part of the sheet, and
    # dividing those by the whole of it digs a well there. The floor under the
    # lanes is prof.min() (see the load filter below), so a well at one edge is
    # not a cosmetic edge effect -- it drops the floor by a third of the ink,
    # every bare-paper cell then clears 0.25 of the median load, and a scan
    # with junk beside its field came back cut into 95 lanes of a printed 59.
    hits = np.zeros(n)
    for s in np.unique(np.clip(shift, -n, n)):
        at = shift == s
        band, k = ink[at].sum(0, dtype=np.float64), at.sum()
        if 0 <= s < n:
            prof[:n - s] += band[s:]
            hits[:n - s] += k
        elif -n < s < 0:
            prof[-s:] += band[:n + s]
            hits[-s:] += k
    return prof / np.maximum(hits, 1)


def find_pitch(ink, x0, x1, min_pitch=6, rows=512, parts=3, subharm=True,
               lean=0.0):
    """Lane pitch in px, from the row-wise mean FFT magnitude.

    Averaging the column profile first and transforming that only survives
    while every lane's ink sits at the same offset: a fixed left wall
    does, a curve's wandering stroke does not, and the smeared profile loses a
    lane (119 of 120 printed). Transforming each row and averaging the
    MAGNITUDES keeps the grating: the phase differs row to row, the period does
    not. This is how get_wav.m counts grooves.

    The peak bin is refined by the vertex of a parabola through the log of its
    two neighbours, on a Hann-windowed row -- a whole bin is half a pitch of
    grid drift by the right edge of a 121-lane page, which is what decides
    whether the last lane is found. Without the window the parabola inherits
    the leakage of a span that is not a whole number of lanes and reads 39.08
    where the page was printed at 39.00, which is that same half pitch back.
    """
    a = ink[:, x0:x1].astype(np.float64)
    lit = a.sum(1)
    a = a[lit > lit.max() * INK]
    a = a[:: max(1, len(a) // rows)]     # pitch is one number; 512 rows fix it
    n = a.shape[1]
    if n < 2 * min_pitch:
        # Not wide enough to hold two lanes, so there is no grating to measure
        # and no bin to point at -- a page with one speck of ink on it lands
        # here. An infinite pitch fails find_tracks' own bounds check, and the
        # ink is read as the single lane it is.
        return float("inf")
    F = np.abs(np.fft.rfft((a - a.mean(1, keepdims=True)) * np.hanning(n), axis=1))
    mag = F.mean(0)
    F = np.log(mag + 1e-12)
    k = 2 + int(np.argmax(F[2:max(3, n // min_pitch)]))

    # A grating whose bars MOVE can beat down its own fundamental. Every lane
    # holds a different second, so past about half the lane's swing in rms the
    # strokes wander far enough that the bin at the true pitch is no longer the
    # tallest, and the peak lands on twice or three times it. That is not a
    # near miss: the page is then cut into two or three times too many lanes,
    # every second after the first is shifted, and nothing downstream notices.
    #
    # So a bin is not believed until a half and a third of it have been asked.
    # Whichever of those still carries SUBHARM of the winner is the better
    # fundamental, and the LOWEST such is taken -- on a page that landed on the
    # third harmonic both k/2 and k/3 answer, and only k/3 is the pitch.
    #
    # Asked ONCE. Run as a loop it walks away: past the point where the
    # fundamental is beaten the whole low end of the spectrum is smeared
    # upwards, so every further halving passes as well, and a page of 20 lanes
    # comes back as 5 at four times the pitch -- which is worse than the two
    # harmonics it was meant to fix, because it is wrong in the safe-looking
    # direction.
    #
    # The threshold is not delicate. Measured as the ratio of the subharmonic
    # bin to the winner, on pages whose peak IS the fundamental and on pages
    # where it is not:
    #
    #     fundamental found   04.png 0.025   sheet.png 0.015   up to 0.245
    #     fundamental beaten  0.642   0.733   0.921   0.948
    #
    # Real audio is normalised to a peak and does not reach this: out3.wav
    # prints at 0.071 and has to be hard-limited four times over before it
    # crosses. A test signal walks straight into it.
    # Not asked below four periods across the page. That far down the bins are
    # too coarse and the window's own leakage too wide for the question to have
    # an answer: a sheet of four lanes has its peak at bin 4, bin 2 carries the
    # envelope rather than a grating, and the page halves itself into two. Four
    # lanes is the smallest page here that this test leaves alone and the
    # smallest it needs to.
    if subharm:
        lower = [j for j in (int(round(k / 2)), int(round(k / 3)))
                 if j >= 4 and mag[max(j - 1, 0):j + 2].max() >= SUBHARM * mag[k]]
        if lower:
            k = min(lower)

    if 0 < k < len(F) - 1:
        d = F[k - 1] - 2 * F[k] + F[k + 1]
        if d < 0:
            # Half a bin is as far as a vertex through three samples can
            # honestly sit from the middle one; past that the "peak" is a slope
            # and the formula runs away -- on a page with a single lane it
            # walked the bin past zero and handed back a NEGATIVE pitch, which
            # left the cut search with nothing to search and killed the read.
            k += np.clip(0.5 * (F[k - 1] - F[k + 1]) / d, -0.5, 0.5)
    pitch = n / k
    if not subharm:
        # A third of a page measures; it does not get an opinion about
        # harmonics. Its window is ten lanes wide, and down there the bin a
        # half or a third of the winner points at is not a grating at all but
        # the envelope of the ink across the window -- which passes SUBHARM
        # comfortably (0.74 to 0.93 measured, against 0.33 to 0.39 for the same
        # sheet asked whole) and drags the pitch up by half again. That is how
        # a 30-lane page came back as 22.
        return pitch

    # Whether that is the pitch or a harmonic of it is a question the paper can
    # answer and the spectrum cannot; see harmonic().
    prof = col_profile(ink[:, x0:x1], lean)
    pitch = harmonic(prof, pitch)

    # A page's own pitch is not one number. Measured on the four scans of
    # 26.08, thirds of the same sheet come back 37.5907 / 37.5975 / 37.8176 --
    # 0.6% wider at one end, the same 0.6% on every sheet of that session and
    # 0.0000% on the render it was printed from, so it is the printer and the
    # flatbed rather than the format.
    #
    # One number for the whole page then sits BELOW every local one (37.5775
    # against all three above), the grid runs faster than the paper, and by the
    # far edge it has gained a lane. On a 38.7 px sheet the drift is a quarter
    # of a lane and nothing notices. On the 19 px sheet it is HALF a lane: the
    # read came back with an extra lane, a 7 px sliver at the edge, every later
    # second shifted by half a second, and 13 seconds of noise where the sliver
    # fell. The median of thirds is enough to fix it -- the same sheet reads
    # back at its printed 240 lanes.
    #
    # Not asked below eight lanes to a third. Down there the bins are too
    # coarse and the window's leakage too wide for the question to have an
    # answer, and the whole-page number is the better one anyway: a page that
    # small has no room to drift.
    if parts > 1 and (x1 - x0) >= parts * 8 * pitch:
        w = (x1 - x0) // parts
        med = harmonic(prof, float(np.median(
            [find_pitch(ink, x0 + i * w, x0 + (i + 1) * w, min_pitch, rows,
                        parts=1, subharm=False, lean=lean)
             for i in range(parts)])))
        # And the thirds do not get the last word either. Three windows that
        # agree with each other and not with the page have not measured the
        # page's drift; they have made the same mistake in the same narrow
        # window, and two of the three carry the median. The drift this is here
        # to catch is 0.6% of a pitch. Past VETO it is the other thing.
        if abs(med / pitch - 1) <= VETO:
            pitch = med
    return pitch


def odd_lane(lanes):
    """What to say about a lane that is not one, or None if they all are.

    A lane that comes out half the median width is something else on the sheet
    being segmented as if it were a lane -- a mark printed beside the field, a
    punch hole, the scanner's own edge. It is worth saying out loud because
    nothing else notices: the page comes back one lane longer than it is and
    every second after the intruder is shifted by one, and it still sounds like
    the recording. Measured narrowest lane against the median: 0.95 on the
    crooked sheet, 0.77 on the rippled one, and 0.22 on a scan carrying a
    printed mark.

    Said by both readers, from here, so that the wav and the folder of strips
    cannot end up disagreeing about which sheets are worth a warning.
    """
    w = np.array([b - a for a, b in lanes], float)
    if len(w) > 2 and w.min() < 0.5 * np.median(w):
        return (f"lane {int(np.argmin(w))} of {len(w)} is {w.min():.0f} px "
                f"against a median of {np.median(w):.0f} -- something on this "
                f"sheet is being segmented as a lane that is not one, and "
                f"everything after it is one lane out")
    return None


def cut_grid(prof, pitch, stat=np.mean):
    """The grid of this pitch whose cut lines carry the least ink: (ink, cuts).

    One period is slid across and the offset whose cut lines are cleanest is
    kept. Taking the phase of the same FFT bin would be shorter but biases by a
    fraction of a pitch, because a lane is not symmetric (fixed left edge,
    modulated right one).

    Costed against a profile smoothed over a quarter pitch, so a cut hugging
    the edge of a gap is charged for the ink just beside it. On bare prof the
    gaps read as a flat zero plateau and the cut lands wherever argmin happens
    to break the tie, a pixel away from the next lane.
    """
    n = len(prof)
    w = max(1, int(pitch / 4))
    near = np.convolve(np.pad(prof, w, mode="edge"), np.ones(w) / w, "same")[w:-w]
    grid = np.arange(-1, n / pitch + 1)
    cuts = [o + pitch * grid for o in np.arange(0, pitch, 0.25)]
    ink_on_cut = [stat(np.interp(c[(c >= 0) & (c <= n - 1)], np.arange(n), near))
                  for c in cuts]
    i = int(np.argmin(ink_on_cut))
    return ink_on_cut[i], cuts[i]


def harmonic(prof, pitch, tol=CUTS):
    """The pitch, or the multiple of it the page is actually cut on.

    A grating whose bars move beats down its own fundamental and the peak lands
    on twice or three times the pitch. SUBHARM asks the spectrum about that and
    cannot always be believed: the ratio it tests reaches 0.245 on pages whose
    peak IS the fundamental and falls to 0.235 on pages where it is not, so the
    two populations overlap and no threshold lies between them.

    The paper is not ambiguous about it. At half the pitch every other cut line
    falls in the middle of a lane and crosses its stroke; at the pitch they all
    fall between lanes. Measured on a 120-lane page at 0.9 of full swing, where
    the spectrum says 19.34 and SUBHARM lets it pass: the cut lines cost 12.3
    there against 8.3 at 38.69, and the page is cut into its printed 120.

    The SMALLEST candidate that is nearly as clean as the cleanest wins, not
    the cleanest -- when the pitch is right, the cuts of twice it are a SUBSET
    of its own and score just as well, so cleanest is a coin toss between the
    two and smallest is the answer. Only twice and three times are asked: a
    peak four lanes out is not something this has been seen to do, and every
    further candidate is another chance to be talked out of a pitch that was
    right.
    """
    cands = [pitch * m for m in (1, 2, 3) if pitch * m <= len(prof) / 2] or [pitch]
    # The MEDIAN of the ink on the lines, not the mean: one line landing on
    # something solid -- the scanner's own edge past the end of the field, a
    # mark printed beside it -- is worth 15 of mean ink here and decides the
    # question on its own. Half the lines have to be dirty before a median
    # moves, and half the lines dirty is exactly what a halved pitch looks
    # like. (find_tracks keeps the mean for choosing the OFFSET, where the
    # question is which of two grids a pixel apart is cleaner, and every
    # candidate carries the same solid edge anyway.)
    costs = [cut_grid(prof, c, np.median)[0] for c in cands]
    best = min(costs)
    return next(c for c, s in zip(cands, costs) if s <= tol * best)


def find_tracks(ink, pitch=None, min_pitch=6, lean=None):
    """Track boundaries as (x0, x1) column slices, left to right.

    The page is a grating, so the pitch comes from find_pitch(). Where to cut
    is then chosen by cut_grid(), which is also how find_pitch settles whether
    it found the pitch or a harmonic of it.

    Splitting on runs of white columns instead only works while the lanes never
    swing into the gaps. Pack the tracks tighter and they do, the profile never
    reaches paper, and the whole page reads as a single track.

    The profile is summed along the ink's own lean, not straight down -- see
    col_profile(). That is the only place on the page skew ever broke: pass
    `lean` to override the measurement, 0.0 to sum straight down as this used
    to.
    """
    if lean is None:
        leans = ink_lean(ink)
        lean = min(leans, key=abs) if leans else 0.0
    prof = col_profile(ink, lean)
    x0, x1 = inked_span(prof)
    p = prof[x0:x1]
    n = len(p)

    if pitch is None:
        pitch = find_pitch(ink, x0, x1, min_pitch, lean=lean)
    if not np.isfinite(pitch) or not min_pitch <= pitch <= n:
        # No grating the estimate can believe in: read the ink as one lane
        # rather than slicing it into nonsense. A sheet with a single lane on
        # it is the honest case -- the search runs from two periods up, so one
        # period is not an answer it can give.
        return [(x0, x1)]

    edges = cut_grid(p, pitch)[1] + x0

    edges = np.clip(np.round(edges).astype(int), 0, ink.shape[1])
    lanes = [(a, b) for a, b in zip(edges, edges[1:]) if b - a >= min_pitch]

    # Keep the lanes that carry ink, counted above the paper floor -- clipped
    # sensor noise lifts bare paper off zero, and at 15/255 the margins would
    # otherwise pass for lanes of their own. The grid is laid a period wide on
    # either side on purpose, so the empty overshoot has to go; testing instead
    # whether a lane's centre falls inside the inked span drops a real lane
    # whenever its stroke keeps to the far side of it -- which is exactly what
    # the leftmost lane of a curve page does, and it cost a lane, and every
    # lane after it, on a bitonal scan.
    #
    # Too heavy is not a lane either, and that bound is the tighter of the two.
    # Every lane carries the SAME ink by construction -- a stroke of fixed
    # width down a fixed number of rows, however far it wanders -- so the
    # spread is narrow enough to bound: 0.82 to 1.10 of the median across five
    # printed and scanned sheets here and two straight out of render_page. The
    # scanner's own edge, segmented at the right pitch because a grating does
    # not care what it is measuring, sits at 1.8 to 5.7 of it. One of the six
    # scans here carried four such lanes past its right-hand clock, which cost
    # the read four seconds of nothing on the end, its second clock, and its
    # retiming from both ends -- in silence, but for the narrow-lane warning
    # that did not apply.
    floor = np.maximum(prof - prof.min(), 0.0)
    load = np.array([floor[a:b].sum() for a, b in lanes])
    mid = np.median(load)
    return [t for t, l in zip(lanes, load) if 0.25 * mid < l < 1.4 * mid]


def ink_rows(win, frac=INK):
    """First and last row of a lane's own ink, or None if it carries none.

    Counted as pixels that are actually inked, not as the row's sum: a blank
    row of scanned paper SUMS to a couple of percent of an inked one, so a sum
    test passes every blank row and measures the lane as tall as the page. The
    lanes of a real scan do not all start on the same row -- on an early test
    scan the top edge of the ink walks 23 rows across the sheet while the
    bottom edge stays put -- and a lane read from the page's rows instead of
    its own is offset in TIME by the difference. It cost the whole read: 0.30
    against the printed audio, 0.81 once each lane is trimmed to its own ink.

    Then bounded by the runs that are long enough to BE a lane, because a
    threshold this low is otherwise decided by any single blot: a speck of dust
    300 rows above the ink is one lit row, and the lane is stretched to reach
    it. A lane's ink is one run the height of the page; a speck is a few rows.
    """
    strong = (win > ink_level(win)).sum(1)
    lit = strong > strong.max() * frac
    edge = np.diff(np.r_[False, lit, False].astype(np.int8))
    runs = [(a, b) for a, b in zip(np.flatnonzero(edge > 0), np.flatnonzero(edge < 0))
            if b - a > len(win) // 100]
    return (runs[0][0], runs[-1][1]) if runs else None


def resample(area, n):
    """Stretch a lane to exactly n samples.

    Tracks differ by a few rows in length, so each is stretched to the same n;
    without that the joins click once a second.

    Nothing is cropped first, on purpose. Print and scan fade a stroke's first
    and last rows into the paper, but those rows are samples: a guard that
    dropped them cost one sample from every lane, swept as inert up to 4 rows
    and collapsed the read at 8. A knob with no value between "keep them" and
    "broken" is not a knob, so it is gone.
    """
    return np.interp(np.linspace(0, len(area) - 1, n), np.arange(len(area)), area)


def highpass(x, win):
    """Subtract a moving average: removes the paper's own slow wander and any
    drift in print density across the page.

    Done on the joined signal, not per track: a per-track mean would kill real
    low frequencies and leave a step at every join.
    """
    win |= 1  # odd, so 'valid' returns exactly len(x)
    pad = np.pad(x, win // 2, mode="reflect")
    return x - np.convolve(pad, np.ones(win) / win, "valid")


def declick(x, joins, n):
    """Spread the step at each join over n samples; the format's own default
    lives in paper_sound.DECLICK.

    Each track is an independent second, so the waveform does not line up
    across a join and there is no overlap to cross-fade. Tilting the two sides
    together over a couple of milliseconds -- 2.4 of them at the format's own
    16 samples on an A4 sheet at 600 dpi -- turns a broadband click into an
    inaudible low bump.
    """
    m = n // 2
    if m < 1:
        return x
    x = x.copy()
    ramp = np.arange(1, m + 1) / m
    for j in joins:
        if j - m < 0 or j + m > len(x):
            continue
        half = (x[j] - x[j - 1]) / 2
        x[j - m:j] += half * ramp
        x[j:j + m] -= half * ramp[::-1]
    return x


def full_scale(x):
    """What to divide `x` by so that its loudest fills a lane. 0 for silence.

    The one rule, in one place: print, read and the slip bench all put full
    scale at the 99.9th percentile with HEADROOM over it, and a sheet printed
    by one number and read back by another is off by the difference.
    write_wav below is where that number is measured and argued.

    A percentile is not a maximum, and that is the whole of this function. On
    audio that is silence with a few samples of sound in it -- a click track,
    a tail of digital black, anything sparse -- 99.9% of the samples ARE zero,
    so the percentile is zero, and every caller here read a zero scale as "all
    silence" and threw the signal away. Ten samples of 0.8 in 16000 printed
    the same page as a file of pure silence, and read back as one. So the peak
    is the floor under the rule: it is used only when the percentile cannot
    see the signal at all, it can only ever make a file louder than the old
    behaviour rather than quieter, and it is zero only when the signal is.
    """
    x = np.abs(np.asarray(x, float))
    if not x.size:
        return 0.0
    p = float(np.percentile(x, 99.9))
    return (p if p else float(x.max())) * HEADROOM


def write_wav(path, x, sr):
    # Scaled by a high percentile, not by the peak. One lane the reader lost
    # swings into its neighbours at three times the legal excursion, and under
    # peak normalisation that one lane sets the level for the whole file:
    # measured on the 240-lane sheet, 27 lost lanes out of 240 put the other
    # 213 eleven dB down, which is heard as "the sheet came out quiet" and is
    # nothing of the sort.
    #
    # Where to cut is not a matter of taste. The ratio of the peak to the
    # 99.9th percentile separates the two populations by a factor of three:
    #
    #     healthy reads   1.19  1.21  1.31  1.46      source audio  1.22
    #     a read with 27 lost lanes           3.63
    #
    # so HEADROOM sits between them. At 1.3 every healthy file above clips
    # 0.000% of its samples -- the source wavs included, which is the check
    # that this is not clipping music -- and the broken one clips 0.026%,
    # being the excursions that were never audio. All of them then land within
    # 0.6 dB of the level that was printed.
    # A read that is all silence -- a page lost whole, a tail of blank lanes --
    # has no percentile to scale by, and dividing by it makes the silence NaN.
    # It writes as zeros either way here; this is so that it does so on purpose.
    scale = full_scale(x)
    x = np.clip(x / scale * 0.98, -0.98, 0.98) if scale else np.zeros_like(x)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def cumsum_at(cs, pos):
    """Cumulative sums read at fractional column positions, one per row."""
    i = np.clip(np.floor(pos), 0, cs.shape[1] - 2).astype(int)
    f = np.clip(pos - i, 0, 1)
    rows = np.arange(len(cs))
    return cs[rows, i] * (1 - f) + cs[rows, i + 1] * f
