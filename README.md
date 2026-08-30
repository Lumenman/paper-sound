# paper_sound

Print audio onto paper as a wandering line, scan the paper, get the audio back.

Nothing is encoded anywhere. **One lane is one second, one row is one sample**,
so the sheet's printable height in pixels *is* the sample rate and its width is
how many seconds fit. Pitch, rate, lane count and skew are all measured off the
sheet when it is read back — there is no header, no checksum, no metadata. A
sheet is a picture of its own contents.

One A4 sheet at 600 dpi holds **120 seconds at 6780 Hz**, plus two clock lanes.

```
python paper-sound/paper_sound.py print song.wav -o sheet    # -> sheet_01.png, sheet_02.png, ...
python paper-sound/paper_sound.py read sheet_01.png -o back.wav
python paper-sound/paper_sound.py selftest
python paper-sound/bench.py sheet_01.png song.wav            # score a read against the original
python paper-sound/bench.py --same a.png b.png c.png song.wav  # one sheet scanned thrice: the median
```

A 120-second recording, on the defaults:

```
$ python paper-sound/paper_sound.py print song.wav -o sheet
song.wav: 120 s at 6780 Hz, 120 per sheet -> 1 sheet
  a4 210x297 mm = 4961x7016 px at 600 dpi, margin 5 mm, pitch 38.7 px = 1.64 mm, clock yes
  sheet.png: 120 lanes, seconds 0-120, ink 4712x6780 px = 199x287 mm, margins 5.2 mm sides, 5.0 mm ends

$ python paper-sound/paper_sound.py read sheet.png -o back.wav
sheet.png: 2 clock lanes found, carriage ripple 0.07 samples rms
sheet.png: 4961x7016 px, 120 lanes, 6780 Hz
wrote back.wav: 813600 samples, 120.0 s at 6780 Hz
```

Lane counts are seconds of sound throughout: the two clocks are printed and
used, and counted by neither side.

Needs `numpy` and `pillow`.

The program lives in `paper-sound/`, and so do the Russian docs: a first-time
guide (printing, scanning, calibration, testing, the scanner's whine) in
`paper-sound/GUIDE.ru.md`, the same format description as this file in
`paper-sound/README.ru.md`, and the lab journal -- every number and every dead
end -- in `paper-sound/LAB.ru.md`. This file is the format itself.

## The carrier

A lane carries its second as a stroke of fixed width wandering inside it. The
sample is the stroke's **position**, read as the ink's centroid across the lane.

Position rather than area, because a centroid is measured against the lane's own
centre and is blind to how much ink landed — print density, exposure and gamma
drop out of the answer. That blindness is the whole reason this shape was
chosen, and it holds **only while the scanner's response is linear**, which is
why the scanning rules below are not a matter of taste.

## What the carrier does to the band

Nothing. A row is not a slit — it does not integrate the sound over its own
duration, it holds one stroke position, which is one sample. A sheet printed and
read back without ever touching paper returns what went in, **flat to 0.05 dB**
from nothing to Nyquist.

Paper is where the top goes, and it does not go the way a filter takes it. The
stroke has to travel `A·2πf/sr` pixels from one row to the next, so speed costs
amplitude and frequency together; ink spreads and glass blurs, both **across
rows**, and rows here are time. While the step per row stays under the stroke's
own width the neighbours overlap and the blur only smooths the position. Once it
does not, the blur mixes two positions that have nothing to do with each other.

So the ceiling is a **slew limit, not a bandwidth**, and it behaves like one.
Measured on three sheets differing only in how much top was printed: at 2600 Hz
+4.3 dB of extra ink came back as +0.7, and at 3000 Hz +6.1 dB came back as
−0.2, while below 1200 Hz the same sheets pass their differences through one for
one. Pre-emphasis buys the first few dB and nothing after — `prep.bat`'s `+4` at
2600 Hz was chosen by ear and sits on the knee, and `prep4.bat` is the record of
what happens past it.

What does buy top is `--rows 2`. It halves the step per row for the same audio,
which lifts the ceiling by 6 dB; its measured **+2.1 dB** had no explanation
until this, and the two-row sheet here loses half as much at 2600 Hz as the
one-row sheets do. Lifting the top on **read** buys nothing at all — the paper's
noise is in the same band as the signal, measured at −0.11 dB on one sheet and
+0.06 on another over eight settings of an inverse filter.

## Printing and scanning

Print at 100%, actual size, no "fit to page" — the PNG carries its own dpi. If a
sheet comes out faint, print it again darker; do not fix it afterwards.

Lay the sheet against the guide, though the budget for getting it wrong is now
the glass rather than the reader: the grid is cut on a profile summed along the
sheet's own lean, so a synthetic sheet sheared **16 degrees** still reads back
at r=1.0000. What runs out first is the platen -- A4 turned on A4 glass stops
fitting at about 1.2 degrees, its ink block alone at about 3. The scans here
lay at 0.02 to 0.31 degrees. Until the profile followed the lean the limit was
0.33 degrees, one lane pitch, and what failed there was the cut.

Scan greyscale at the same dpi with **every adjustment off**: no auto-contrast,
no levels, no white or black point, no sharpening, no deskew, not bitonal.
8-bit or 16-bit, either reads the same -- the depth is scaled on the way in and
every number here was measured at 8.

Setting a white point is the expensive one. It costs **8.6 dB, in silence, and
cannot be undone**. Zeroing everything paler than 4 parts in 255 is enough to do
it. The pixels that operation throws away are the partial-coverage ones at the
stroke's edges, and those are the whole of the sub-pixel position: a 4 px stroke
sitting at x.3 differs from one at x.7 only in how its two edge pixels share the
ink. It bites hardest when the print came out light, because then the white
point reaches into the ink instead of into the paper.

`read` says when it sees the signature of a tone curve:

```
sheet.png: levels have been adjusted after scanning (128 empty histogram
bins). That may have cost nothing, or it may have cost 8 dB -- nothing in the
image says which. If this read is disappointing, rescan with every adjustment
off
```

A scanner's own output fills its histogram; stretching it spreads the same
values over a wider range and leaves a comb of empty bins. Measured: 0 empty
bins on a sheet straight out of `print`, 4 on a raw scan, 128–133 on retouched
ones. The note cannot say whether the retouching *hurt* — three ways of
measuring that from the image alone were tried and all three failed, one of them
ranking the ruined scan cleaner than the intact ones. It reports what it found
and what to do.

## What the two corrections are worth

Both were measured the same way on two printed and scanned sheets, and the two
sheets disagree about which one matters:

| stage | the crooked sheet | | the rippled sheet | |
|---|---|---|---|---|
| raw read | 0.629 | −1.84 dB | 0.839 | +3.77 dB |
| + skew estimate | 0.727 | +0.50 dB | 0.842 | +3.86 dB |
| + clock retiming | 0.781 | +1.94 dB | 0.939 | +8.74 dB |
| | | | | |
| **skew is worth** | | **+2.34 dB** | | **+0.09 dB** |
| **the clocks are worth** | | **+1.44 dB** | | **+4.88 dB** |

Neither number is a property of this reader. The first sheet lay 17.7 px crooked
with a well behaved carriage; the second lay 4.9 px crooked with a carriage
rippling 9 samples peak to peak. Each correction is worth exactly as much of its
own defect as the sheet happened to carry, and neither defect is knowable before
the sheet is on the platen.

So read the table as insurance rather than as gain. The premium is fixed and
small — the skew estimate costs nothing and needs no mark on the page, the
clocks cost two lanes, 1.6% of the sheet — and the payout is whatever that day's
scan happened to need. On the evidence so far that runs from 0.1 dB to 5 dB for
either one. Keeping both is justified by the worst sheet, not by the average.

**The skew estimate** is one line fitted to the mean of every lane's centroid:
each lane holds a different second, so content cancels as the square root of the
lane count and geometry survives.

**The clock lanes** buy the one thing the sheet cannot otherwise give up. A
scanner does not advance one row per row, and its carriage ripple is a timing
error shared by every lane — purely vertical, so it leaves no horizontal trace
and no other measurement on a finished sheet can see it. The two are printed
at different rates — 32 samples a cycle on the left, 20 on the right — which
breaks the sheet's one symmetry: which clock reads back first says which way
up the sheet went on the glass, so an upside-down scan turns itself.

The two clocks are not averaged. They disagree — 0.61 raw, 0.82 once each one's
own straight line is removed — and the disagreement is a slope: −1.24 samples
across the left clock's lane against +1.63 across the right. A carriage rippling
vertically cannot do that; the sheet's own geometry can. So each lane is retimed
off its own point on the line between the two, which is worth 0.14 dB over the
average and needs no choice about which end to trust.

## Options

### `print`

| option | default | what it does |
|---|---|---|
| `audio` | | the 16-bit PCM wav to print |
| `-o, --out` | `<audio>.png` | output name; becomes `<out>_01.png`, `_02`… across sheets |
| `--paper` | `a4` | `a4/a3/a5/letter/legal`, or `WxH` in mm |
| `--dpi` | `600` | print resolution — and therefore the sample rate, since the printable height in pixels is it |
| `--margin-mm` | `5` | paper the printer cannot reach, per edge |
| `--pitch` | `38.7` | lane pitch in px, 1.64 mm at 600 dpi. Sets how many seconds a sheet holds: 120 on A4 at the defaults. Under 9 px the stroke has no room left to swing in, and printing refuses rather than laying the sound down inverted |
| `--rows` | `1` | rows of paper per audio sample. At `2` the stroke moves half as far per row, a lane holds half a second, and a sheet holds half the audio: half the paper for **+2.1 dB** (measured on paper with this knob: the median of four scans of one sheet, 12.98-14.93 dB against 11.78 for the control on the same audio). The clocks carry the number themselves -- their period is printed in samples and comes out twice as long on paper -- so `read` is never told, and has no flag for it |
| `--start` | `0.0` | seconds to skip before the first sheet |

The clocks are not optional any more: their own table above prices them at
1.6% of the sheet for 1.4–4.9 dB, and no sane run declines that.

### `read`

Everything here is measured off the sheet. These are recovery knobs for when
that goes wrong, plus one real tuning knob.

| option | default | what it does |
|---|---|---|
| `scans` | | one or more sheets, in playing order |
| `-o, --out` | `<first scan>.wav` | output wav |
| `--pitch` | measured | force the lane pitch in px, if segmentation missed |
| `--upside-down` | off | turn every scan 180° before reading — only for sheets that cannot say which way up they lay, see below |
| `--rows` | measured off the clocks | rows of paper per audio sample. Exactly one sheet needs it: one printed with `--rows 2` whose clocks are damaged, because nothing else on a sheet carries the number. Thinning takes **both** clocks agreeing: reading one row as two costs the top half of the band (32.64 dB against 12.49) and the wav will not give it back, while reading two as one costs a `sox speed` and nothing else, so a lone clock buys the cheap mistake and names the flag |
| `--sr` | measured | force samples per lane, if sheets differ in height |
| `--aperture` | `8` px scaled by the sheet's pitch | px of lane the centroid looks at. **The one knob worth turning** — see below |
| `--no-pilot` | clocks used | read the clock lanes as audio, giving up the retiming: 0.781 → 0.727 on one sheet, 0.939 → 0.842 on the other |
| `--despeckle` | `0` (off) | flatten samples that jump more than this multiple of the median step. For a filthy page only — see below |

The knobs that used to sit here — `--sticky`, `--hp`, `--declick` — were swept
over a real sheet and moved r by 0.0002 to 0.002. Their mechanisms are real,
so they live on as constants at the top of `paper_sound.py`; anyone
re-measuring them edits the constant. `--guard` is gone entirely: cropping
rows off a lane's ends was inert up to 4 and collapsed the read at 8, and a
knob with no value between "keep them" and "broken" is not a knob.

#### `--upside-down` is for sheets that predate the two clock rates

A sheet used to be symmetric: turned 180 degrees it was still a block of
evenly spaced lanes with a clock at each end, and nothing on it said which end
was the top. Now the two clocks are printed at different rates, so which one
reads back first says which way up the sheet lay, and `read` turns an
upside-down scan by itself, per sheet, saying so when it does — a stack where
only some sheets went on the glass backwards reads in one run. One old sheet
here, scanned upside down, scored 0.003 as it lay and 0.9455 turned; a new
sheet turns itself before that can happen.

The flag remains for a sheet that cannot say: printed without clocks, or
printed before the rates differed. **Such a sheet now names itself:** finding
the same rate at both ends, `read` prints that this sheet cannot say which way
up it lay and offers the flag. Staying quiet here is not an option -- it is the
one case where a sheet plays backwards and no measurement anywhere reports a
fault. `04.png` in this set is exactly that sheet: 0.0029 as it lay, 0.4706
turned. It turns the array, not the file -- the
scan on disk stays the original. A 180 degree turn is exact and lossless on
pixels, so a sheet read this way scores what it would have scored the right
way round: **0.9995 against 0.9994** on the same sheet here. Get it wrong in
either direction and the read is obvious rubbish, r around 0, not a subtly
worse version of the recording. The flag applies to every scan in the list.

#### `--aperture` is the only one whose best value is not universal

Measured on two sheets, and the optima point in opposite directions:

| aperture | never printed | printed and scanned |
|---|---|---|
| 3 px | +13.00 dB | **+1.95 dB** |
| 4 px | +16.69 dB | +1.92 dB |
| **8 px** (default) | **+31.10 dB** | +1.80 dB |
| 16 px | +31.04 dB | +1.77 dB |

A wide window sees all of the stroke; a narrow one rejects the grime beside it.
Where the trade lands depends on how dirty the scan is, which only the person
holding it knows.

**The default stays wide because the loss is not symmetric.** Too narrow on a
clean read costs 18 dB; too wide on a dirty one costs 0.15 dB. Try
`--aperture 3` on a visibly noisy scan and leave it alone otherwise.

**The window follows the pitch.** `APERTURE` is tuned at 38.7 px, and what it
has to hold is the smear one row of a moving stroke leaves -- which grows with
the lane, because the excursion does. The default is therefore
`8 * pitch / 38.7`, with the pitch measured off the sheet itself. Optima taken
on three geometries: **4 px at pitch 18.4, 8 at 38.7, 16-24 at 77.4.** Reading
the wide sheet at 8 costs **6.0 dB**; on sheets at the standard pitch the rule
changes nothing (37.4 px asks for 7.7 px of window instead of 8, and every
score holds to four figures). A number given on the flag is used as it stands.

#### `--despeckle` is off for a reason, and on for a reason

On an ordinary scan every setting is worse than off, and the cheapest is
ruinous:

| off | 2 | 3 | 5 | 8 | 12 |
|---|---|---|---|---|---|
| 0.781 | 0.122 | 0.290 | 0.474 | 0.614 | 0.684 |

A printed stroke at a real transient steps as hard as a speck of dust does, and
nothing in the trace tells them apart. But on a page that is genuinely filthy it
is the difference between a read and no read — measured with specks of solid ink
pasted onto a sheet:

| blots on the sheet | off | at 20 |
|---|---|---|
| none | 0.811 | 0.753 |
| 200 of r=4px | 0.786 | 0.742 |
| 2000 of r=4px | 0.559 | **0.583** |
| 2000 of r=8px | 0.323 | **0.476** |

It is a recovery knob, not a tuning one: leave it off until a sheet is visibly
dirty, then it is the only thing that helps.

## Known limits

- **The limit on crookedness is the size of the glass.** It used to be the lane
  pitch, and it belonged to the cut: lanes are cut as vertical columns, so a
  sheet sheared by more than one pitch put a lane at the top of the page over
  its neighbour's columns at the bottom, and the profile the grid is cut on --
  summed straight down -- smeared into lanes that were never printed. That
  profile is summed along the sheet's own lean now, which is index arithmetic
  and costs no decibels: nothing is resampled, no sub-pixel position is
  quantised. Measured on 6780 rows at a 38.7 px pitch, r against what was
  printed: 42 px of shear read -0.0071 before and 1.0000 now, and so do 240 px
  and 1920 px, the last of which is 16 degrees. The shear is measured off the
  ink's own edges -- a sheared page leans its ink block with it, and that
  measurement owes nothing to what was printed. Past this the ceiling is
  physical: A4 turned on A4 glass stops fitting at about 1.2 degrees and the
  ink block at about 3, so what is needed there is a bigger scanner and not a
  different algorithm. The same edge measurement AIMS the drift estimate's
  first pass, with a fit per strip of rows on top; on paper that is worth
  ±0.003 of r (five sheets up, two down, at most +0.13 dB, on the most bent
  sheet here), and what it is really for is the sheet at 40 px of shear, which
  read back at 0.27 before it and 1.0000 after.
- **The two clocks disagree by a slope**, as above. Riding the line between them
  handles it; what is left is the shared wiggle.
- **Some of the carrier's headroom is still unspent.** A sheet that never left
  the computer reads back at **32.6 dB**; paper reads at **11.8 to 12.3**, and up to
  **14.9** at `--rows 2`. Some
  twenty decibels go between the printer and the glass, and this much of them
  now has a name. **Already won back** by the reader: carriage ripple, 1.4-4.9 dB
  (the clocks); skew, 0.1-2.3 dB (the estimate); the centroid aperture, up to
  6.0 dB on a sheet at a non-default pitch, which is why the window now follows
  it. **Named but not won back:** vertical print/scan blur, measured on paper --
  stepping around it is worth +2.1 dB (`--rows 2`, measured with the knob) at
  the price of half the sheet, which makes it a cost line rather than an exclusion; and the scanner's
  own 823 Hz stripe, +16.6 dB over the source inside the band, which nothing in
  the reader suppresses -- a periodicity of **8.00 scan rows**, which is the
  glass rather than the print: it lands at an eighth of the sheet's paper-row
  rate, so the same stripe sings at 1646 Hz on a `--rows 2` sheet, and another
  scanner will have a period of its own. The remainder is not broken down: impulse noise,
  intensity gamma and plain print and scan grain, in proportions nobody has
  measured.
- A sheet that should carry clocks and has none prints a warning — it was
  printed without them, or the lane grid is miscounted. Finding only one prints
  a warning too, and names the lanes the two clocks sit on: a clock short of the
  edge is the grid having overrun the field, and every lane past it is a second
  of nothing on the end of the read.
- **A lane carries a fixed amount of ink**, so the segmentation bounds it from
  above as well as below. A stroke of one width down a fixed number of rows
  weighs the same however far it wanders — 0.82 to 1.10 of the median across the
  five printed and scanned sheets here and two straight out of `print` — while
  the scanner's own edge, cut at the sheet's own pitch because a grating does
  not care what it is measuring, sits at 1.8 to 5.7 of it. One of the six scans
  here carried four such lanes past its right-hand clock: it read 125 lanes and
  one clock, and reads 120 lanes and two clocks now, at 2.33 samples rms of
  ripple against the 6.06 one end alone reported.
- A lane much narrower than the rest prints a warning: lanes are all one pitch,
  so one at half the median is something else on the sheet being segmented as if
  it were a lane — a printed mark, a punch hole — and every second after it is
  shifted. Measured narrowest against the median: 0.95 and 0.77 on clean sheets,
  0.22 on one carrying a mark beside the field. What the width catches is what
  the weight above does not: a mark carrying a lane's worth of ink in the wrong
  shape.

## Files

| | |
|---|---|
| `paper_sound.py` | the format: print, read, selftest |
| `read_tracks.py` | raster work — loading a page, cutting it into lanes, resampling |
| `bench.py` | scores a read against the audio that was printed |

`paper_sound.py selftest` prints a sheet, reads it back and checks the geometry,
the resampler's anti-aliasing, the skew estimate against a sheet drawn crooked,
the clock against a sheet whose rows have been rippled, the upside-down turn off
the two clock rates, the ink bound that throws out the scanner's own edge,
both warnings above, and the impulse rejection. It touches
no files outside a temporary directory.
