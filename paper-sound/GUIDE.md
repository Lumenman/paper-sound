# Sound on paper: a guide

This program prints audio onto a sheet of paper as curves and reads it back
from a scan. It is not a picture of the sound and not a QR code: **the sheet is
the recording**, and with enough patience you could read it with a ruler.

This guide is for someone opening the folder for the first time: how it works,
what to set, how to print, how to scan, how to check the result, and what to do
when it came out badly. The format and its measurements are described in
detail in `../README.md`; the lab journal with every number is in `LAB.ru.md`
next to this file (in Russian; `ARTICLE.md` tells the story in English).

Every command below is given the way it is run **from the project root** (the
folder above this one):

```
python paper-sound/paper_sound.py ...
```

---

## 1. In five minutes

```
python paper-sound/paper_sound.py selftest                       # is everything intact
python paper-sound/paper_sound.py print song.wav -o sheet        # -> sheet_01.png, ...
                                                                 # print it, scan it
python paper-sound/paper_sound.py read sheet_01.png -o back.wav  # back to sound
python paper-sound/bench.py sheet_01.png song.wav                # how close it came
```

`selftest` prints a sheet in memory, reads it back and checks a couple of dozen
things. If it passes, the install is fine:

```
selftest ok: 10 lanes at 600 Hz, round trip 0.9998, clock 0.800 -> 0.988, ...
```

**What you need:** Python 3, `numpy`, `Pillow`. Nothing else: printing and
reading get by with those two.

Optional: `scipy`, only for `picky.py` (a third-party strip reader); `sox`, only
for preparing audio before printing, and any sox will do, from PATH or your own
copy (§3).

---

## 2. How it works

### One lane is a second, one row is a sample

The sheet is divided into vertical **lanes**. One lane carries one second of
sound. A stroke runs down inside the lane, and **each row of pixels holds one
sample**. So:

- the printable height in pixels **is the sample rate**;
- the width of the sheet is how many seconds fit.

A4 at 600 dpi with 5 mm margins: a printable area of 4724 x 6780 px, a lane
pitch of 38.7 px → **122 lanes = 120 seconds of sound at 6780 Hz** (two lanes
go to the clocks).

### Amplitude is position, not density

A sample is **the stroke's position across the lane**, measured as the ink's
centroid. Not the amount of ink, not its thickness, not its greyness. Because
of that, print density, exposure and the scanner's gamma drop out of the
answer: the centroid is measured against the lane's own centre and is blind to
how much ink landed.

The main scanning rule follows from the same thing: **anything that moves the
levels breaks the carrier**. A white point costs 8.6 dB, silently and
irreversibly (see §5).

### The clocks

The two outermost lanes are not sound but **clocks**: a pure tone printed like
an ordinary second. They carry what the sheet cannot otherwise say:

- **time.** A scanner does not advance exactly one row per row; the carriage
  ripples, and that ripple is shared by every lane. The clocks measure it and
  subtract it, worth +1.4 to +4.9 dB on the sheets here;
- **speed.** How many rows went to a sample (the `--rows` option, §4);
- **which way up the sheet lay on the glass.** The two clocks are printed at
  different periods (32 and 20 samples a cycle), and if they read back in the
  reverse order the sheet lay upside down, and the read turns itself.

### What the sheet says about itself

The file has no header, no checksum and no metadata. Everything is measured off
the scan on every read:

| what | from where |
|---|---|
| lane pitch | the page's periodicity (the grid), plus a median over thirds, since every printer+scanner pair drifts its own way |
| where to cut the lanes | the cleanest column profile |
| sample rate | the height of the ink in rows |
| skew and drift | the ink block's own edge, then a fit over strips of rows |
| rows per sample, orientation, ripple | the clocks |
| exposure, density | not at all: the centroid is blind to them |

So a different scanner changes the numbers the sheet reports, not the code that
reads them.

### Limits of the format

- **A4 at 600 dpi ≈ 6.8 kHz and 2 minutes a sheet.** This is an optical
  soundtrack, not a CD. Longer audio goes on several sheets in a row.
- **No ECC and no header.** A hole in the grid is a second of silence in its
  own place.
- **Full swing closes the gaps between lanes**, so material normalised "to the
  ceiling" reads worse (see §3).
- **Skew** is tolerated as long as the sheet fits on the glass (about 1.2° for
  A4 on A4 glass), not up to some reader setting.

---

## 3. Preparing the audio

The requirements are simple: **mono, 16-bit PCM**. It is better to bring the
rate down beforehand to what the sheet holds (A4/600 → 6780 Hz): then the
reader sees `rate == sr` and does not resample by itself. The program says so
if this was not done:

```
NOTE: song.wav is 44100 Hz and this sheet holds 6780; resampled here through
a 209-tap sinc. `sox in.wav -r 6780 out.wav rate -v 6780` is the same job
```

Ready-made chains for speech sit alongside: `prep.bat`, `prep2.bat`,
`prep3.bat`. They band-limit, compand, clip lightly and resample, and print
`sox --stats` at the end. Inside is a measured density menu with RMS and crest
factor targets.

`prep4.bat` is the fourth one, and it **lost**: do not print with it. It has a
pre-emphasis fitted to the top-end roll-off measured off a scan rather than
chosen by ear, and on paper it came in 1.24 dB behind `prep.bat` (LAB §23.8).
The fit is not to blame: the carrier's top is not attenuated but saturated.
Extra amplitude up top is lost entirely, because the stroke cannot travel far
enough between rows. The `+4 at 2600` in `prep.bat` sits exactly on the knee,
and the ear found that knee before the calculation found the curve. The file is
kept as a record of the experiment and as an example of how to measure: it
writes a `_flat` file next to its output, and such sheets must not be judged
against the printed file, which is distorted on purpose. The more accurate the
pre-emphasis, the worse that comparison looks.

**About sox.** It is only needed here, for preparing audio; neither printing
nor reading asks for it. The bat files first look for their own copy in
`..\sox\` (that is how this project is laid out), and if it is not there they
take `sox` from PATH. Any sox will do: install it system-wide, put it anywhere
and add it to PATH, or just type the same commands by hand. There is nothing
special in them.

The minimum, if you do not want to bother with the compander, is to fix the
format:

```
sox in.wav -b 16 -c 1 out.wav rate -v 6780
```

**About loudness.** Printing normalises the sheet by itself, by the 99.9th
percentile rather than the peak. So a single click in the source **no longer**
eats the swing: it used to take the whole budget and the sheet came out quiet
(measured: −2.1 dB on paper). Turning the final `gain` is pointless, the
program renormalises anyway. But the average level above the paper's noise is
still the compander's business: the carrier gives about 20 dB of
signal-to-noise, and speech arrives with 22 dB of crest factor.

If the audio is compressed so hard that the 99.9th percentile is almost the
peak, printing says so:

```
NOTE: 0.42% of the samples print at the edge of their lane and are flattened
```

---

## 4. Printing

```
python paper-sound/paper_sound.py print song.wav -o sheet
python paper-sound/paper_sound.py print song.wav --paper letter --margin-mm 12
python paper-sound/paper_sound.py print song.wav --rows 2
```

| option | default | what it does |
|---|---|---|
| `--paper` | `a4` | paper size; also `letter`, `a3` or `WxH` in mm |
| `--dpi` | `600` | print resolution; the PNG carries it |
| `--margin-mm` | `5` | margins the printer cannot reach |
| `--pitch` | `38.7` | lane pitch in px. Wider is cleaner, but fewer seconds a sheet |
| `--rows` | `1` | rows per sample, 1–4; speech `2`, music `3` (see below) |
| `--start` | `0` | how many seconds to skip before the first sheet |

`holds` takes the same five geometry options: the same calculation without a
wav, for when you want to estimate what fits before you have the file:

```
$ python paper-sound/paper_sound.py holds 480 --paper a3 --rows 2
a3 297x420 mm = 7016x9921 px at 600 dpi, margin 5 mm, pitch 38.7 px = 1.64 mm, 2 rows a sample
  86.5 s at 9685 Hz a sheet, clocks yes
  480 s of audio -> 6 sheets
```

Seconds and hertz always come together: the printable height in pixels is the
rate, and `--pitch` is pixels too, so `--dpi` moves both at once.

Printing reports what it produced:

```
song.wav: 120 s at 6780 Hz, 122 lanes -> 1 sheet
  a4 210x297 mm = 4961x7016 px at 600 dpi, margin 5 mm, pitch 38.7 px = 1.64 mm
  sheet.png: 122 lanes, seconds 0-120, ink 4724x6780 px = 200x287 mm
```

**Print at 100%, actual size, no "fit to page".** The PNG carries its own dpi,
and a sane print dialog does the rest by itself. Not all are sane: a
dialog that fits the sheet into the printer's printable area shrinks it by
2–5% (the laser here printed 95.2% = 200/210), and the 600 dpi grid is
resampled strokes and all. In IrfanView: "Original size", centred on both axes,
margins 0; without centring the sheet starts at the printable area's corner and
the edge of the block is cut off. Or PDF: `topdf.bat sheet.png` (needs
`pip install img2pdf`) puts the same raster, not recompressed, on an A4 page
with the 600 dpi fixed; print it at "Actual size", not "Fit". Check with a ruler: on A4 the block is
199 × 287 mm. If a sheet comes out faint,
print it again darker, **do not fix it afterwards**: what did not land on the
paper, the reader cannot restore.

**About `--dpi`.** Stroke width, gap and aperture are set in pixels and do not
follow the dpi: at 1200 dpi the stroke becomes 0.085 mm, and an inkjet lays
that down full of holes. Printing knows this and says so, naming the numbers
you need:

```
NOTE: 0.085 mm of stroke, against the 0.17 mm this format has been printed and
scanned at. STROKE 8 and --pitch 77 would keep the sheet the size it is at 600 dpi
```

**About `--rows`.** Each sample is drawn over N rows: the ink's slope per row
is N times smaller, and the sheet holds N times less sound. Measured on two
printers and three recordings (`LAB.ru.md`, §28), dB over `--rows 1`:

```
              voice          harpsichord    an A4 holds
--rows 2      +2.7…+5.2      +4.8…+7.1      60 s
--rows 3      +2.8…+5.8      +6.6…+9.6      40 s
--rows 4      +2.5…+6.5      +6.9…+8.4      30 s
```

The top of each range is the laser printed at 95.2%: its `--rows 1` is pulled
down by the shrink (see above), so the honest gain is nearer the bottom.

**Speech: `--rows 2`.** A third row gives voice +0.1…0.6 dB, less than the
spread between scans, and costs 20 s of the sheet. **Music with treble:
`--rows 3`**, which gives the harpsichord another +2 dB. `--rows 4` gives
nothing beyond the spread.

The clocks carry 1 and 2 themselves, and `read` asks for no flag for them.
**3 and 4 are read only with the same `--rows`**: a clock's period in rows on a
sheet at 3 (60) is 6.7% from one at 2 (64), and printers shrink a sheet by
2–5%, so a reader guessing on its own would read sheets at 2 as sheets at 3.
The reader names a forgotten flag ("both clocks read as a sheet
printed with --rows 3...", §7) but reads the sheet as `--rows 1`, three or
four times as long: read it again with the flag it named. Fractional values (1.5) are
impossible, see §12.

---

## 5. Scanning

**The same dpi, every adjustment off.** No auto-contrast, no levels, no white
or black point, no sharpening, no deskew, **not bitonal**. 8 or 16 bit, either
is fine.

**Greyscale or colour: try both.** On the flatbed here the greyscale mode tops
out at 151 of 255 between paper and ink, while the colour mode reaches 219 in
luma and 254 in red, and that range is exactly the partial-coverage edge the
sub-pixel position lives in. A colour scan costs nothing but three times the
file: the reader reads one channel, and on grey scans the three channels came
back byte-identical.

Why so strict: the carrier is the fraction of a pixel at the stroke's edge,
where the ink covers a pixel partially. A white point zeroes everything paler
than a few parts in 255 and erases exactly those pixels: **−8.6 dB, silently
and irreversibly**. The reader can spot the traces of an adjustment by the
"comb" of empty histogram bins, and says:

```
sheet.png: levels have been adjusted after scanning (37 empty bins ...)
```

But spotting is not fixing. The only answer is to rescan.

Lay the sheet against the guide. Skew is cheap now (the grid is cut along the
sheet's own lean), but the sheet still has to fit on the glass entirely.

**Three scans instead of one.** Two scans of the same sheet, with the same
settings, minutes apart, differed here by **1.95 dB**. Anything smaller than
that is a conversation about the scanner, not about the format. So when you
measure something seriously, scan the sheet three times, lifting it and laying
it down again between scans, and compare medians (`bench.py --same`, §8).

---

## 6. Reading

```
python paper-sound/paper_sound.py read sheet.png -o back.wav
python paper-sound/paper_sound.py read sheet_01.png sheet_02.png -o back.wav
```

Sheets are listed **in playing order**; a second sheet scanned at a different
scale is brought to the first one's rate.

| option | when you need it |
|---|---|
| `--aperture` | the main read knob: the width of the centroid window in px. Turn it first if r is low (§9) |
| `--pitch` | if segmentation missed the pitch |
| `--rows` | always for a sheet printed with `--rows 3` or `4`; for `2` if the clocks are damaged: nothing else on the sheet carries this number |
| `--upside-down` | if the sheet was printed before the clocks became different and plays backwards |
| `--sr` | set the number of samples per lane by hand |
| `--no-pilot` | read without the clocks (and without retiming) |
| `--despeckle` | outlier suppression; **off by default**, and that is a measurement, not caution: on a clean sheet it only hurts (0.781 → 0.122 at threshold 2). Turn it on only for a genuinely dirty sheet |

---

## 7. What reading prints

The reader speaks in the language of the defect. None of these messages stops
the work, the read carries on, but each one means something.

| message | what happened | what to do |
|---|---|---|
| `levels have been adjusted after scanning` | the scan went through a curve or levels | rescan with no adjustments |
| `NOTE: scanned upside down; the read was turned` | the sheet lay upside down, the clocks saw it | nothing, already fixed |
| `both clocks are the 32-sample one` | an old sheet: identical clocks, nothing to tell orientation by | if it plays backwards, `--upside-down` |
| `only one clock` | the second lane is damaged, or the grid overran the field | orientation not confirmed, check by ear |
| `clocks on lanes 4 and 57 of 62 -- the lane grid overran the field` | segmentation caught the edge of the glass | the extra seconds are empty; crop the scan to the paper |
| `no clock lanes found` | printed without clocks, or the grid badly missed | no retiming; `--rows` if the sheet was at 2, 3 or 4 |
| `both clocks read as a sheet printed with --rows 3` | a sheet at 3 or 4 read without `--rows`; the clock warnings below are this | read again with the `--rows` it named |
| `a clock says 2 rows to the sample and that takes two agreeing` | one clock of two was recognised | if the sheet is at 2, `--rows 2` |
| `2 clock lanes found, carriage ripple 1.47 samples rms` | normal: 0.08 on a render, about 1.5 on paper | nothing |
| `lane 3 of 40 is 8 px against a median of 38` | something on the sheet was segmented as a lane | **important**: everything after it is shifted by a second |
| `the drift fitted down this page says ... the ink's own edges say ...` | the drift fit left the sheet, skew read low | rescan it straighter |
| `seconds 12, 13 of 120: no lane on the sheet to read there` | holes in the grid | silence in their place, the tail does not shift |
| `no lane on the sheet to cut at 006` | the same, when cutting strips | the number stays unwritten, strips after it do not shift; there will be a gap in the numbering |
| `stretched to the first sheet's 6606 Hz` | the second sheet was scanned at a different scale | nothing, brought to the first |
| `pHYs carries a CRC its writer computed wrong; mended in memory` | the scanner lied in a metadata block's checksum | nothing, the file on disk is untouched |

---

## 8. Checking: how to measure scans and sound

### `bench.py`, the main instrument

```
python paper-sound/bench.py sheet.png song.wav
python paper-sound/bench.py sheet_01.png sheet_02.png song.wav
python paper-sound/bench.py --same scanA.png scanB.png scanC.png song.wav
```

The last argument is the wav that was printed; everything before it is scans in
playing order. It prints one line:

```
sheet.png   r=0.9383  lanes 57 at 6574 Hz  2 clocks  median 0.9354
            below 0.5: 1/56  worst 0.28 0.79 0.82
            below 0.5 at second 31
```

- **r** is the correlation of the whole read with the original. In decibels:
  dB = 10·log10(r² / (1 − r²));
- **median** is the median of the per-second r. Look at it and at the
  distribution: a reader that lost two seconds out of a hundred barely moves
  the overall r and is very audible;
- **below 0.5** is how many seconds failed. **Where** they are is the
  diagnosis: scattered one by one = defects in individual lanes (look at the
  strips, §11); one long run = the paper (a fold, a band of ripple, the edge
  of the glass);
- **worst** is the three worst seconds. For speech these are often just pauses,
  not failures.

**What to expect.** A sheet straight out of `print`, never on paper:
**r > 0.999**, that is above 27 dB (32.6 here). Less means the reader is
broken, not the paper. Printed and scanned: **0.92…0.985, that is 7.4…14.9 dB**;
the good sheets here read 11.8–12.3 dB, and up to 14.9 at `--rows 2`. The
difference between sheets is the scanner, not the format. **Below 0.85
(4.2 dB) there is something to look for**, and the warnings in §7 are where to
start.

**`--same`** says this is one sheet scanned several times, and adds the median
and the spread. Use it before you believe some change improved anything: the
spread between scans of one sheet reached 1.95 dB here.

### The other instruments

| command | answers the question |
|---|---|
| `python paper-sound/score.py back.wav:song.wav` | the same as bench, but for two finished wavs (the sheet is already read) |
| `python paper-sound/align.py back.wav:song.wav` | **where** the read went off: per-second r together with per-second lag. Growing lag = scale drift, a jump = a lost lane |
| `python paper-sound/spec.py back.wav:song.wav:label` | **what extra** showed up in the spectrum: scanner whine, printer screen (§10) |
| `python paper-sound/slope.py back.wav:song.wav:38.7` | whether the sheet loses **on steepness**: the error against the slope the ink had to draw. Losses only in the top bins = steepness is to blame, flat across all = the paper |
| `python paper-sound/cmp_picky.py 6608 song.wav "reader" back.wav "picky" other.wav` | compare two different reads of one sheet on one metric |
| `python paper-sound/diversity.py scanA.png scanB.png scanC.png song.wav` | what averaging **several scans** of one sheet would buy, and how much error the lanes of one row share |
| `python paper-sound/sigdep.py fitA.png -- testB.png song.wav` | how much of the error is **predictable** from the stroke's shape, and how much of that a reader could take out |

### Self-tests

```
python paper-sound/paper_sound.py selftest      # the format: print -> file -> read
python paper-sound/cut_lanes.py --selftest      # strip cutting
```

Both test real failures, not the happy path: an upside-down sheet, dead clocks,
a symmetric sheet, a broken CRC, a hole in the grid, a sheet skewed by 300 px,
a 16-bit scan. Run them after any change.

---

## 9. A new printer + scanner pair

Generally speaking, **no tuning is needed**: there is not a single number in
the code taken from a particular piece of hardware, and everything that depends
on the hardware the sheet reports about itself (§2). But there are three places
where the pair does show.

1. **`--aperture`** is the only read knob that really depends on the chain's
   blur. The base (8 px at a pitch of 38.7) was measured here and is scaled by
   the sheet's own pitch; a blurrier scanner will want a wider window. If r is
   lower than expected and there are no warnings, sweep it first.
2. **Stroke geometry at print time**: 0.17 mm at 600 dpi. A different dpi or a
   coarse printer screen changes things; printing issues a NOTE and names the
   numbers (§4).
3. **An inverse filter: not worth it.** The print+scan chain acts as a gentle
   low-pass, and the error it leaves does repeat from sheet to sheet: predicted
   from the wav that was printed, it is worth +2-3 dB, and it carries from one
   sheet to another. But the reader does not have the wav. Predicted from the
   read itself, the same correction gives +0.1…+0.8 dB, and below zero when it
   was fitted on another sheet: the treble lies under the paper's noise, and
   lifting one lifts the other (`sigdep.py`, LAB §29.3). Treble can only be
   bought at print time, which is what `prep.bat` does, and `--rows 2`.

**Checklist for the first sheet from a new pair:**

1. print at 600 dpi, 100%, actual size;
2. scan at 600 dpi, greyscale and colour, with **every adjustment off**, and
   keep whichever reads better;
3. `read`, and read the warnings (§7); they are the pair's diagnostics;
4. `bench.py`, and compare against the targets in §8;
5. if it is low, `--aperture` first, everything else after;
6. before fixing anything, three scans and `--same`.

---

## 10. Scanner whine: finding and suppressing it

**Where it comes from.** One row of the sheet = one sample, so **any periodic
pattern in the scanner's rows lands straight in the audio as a tone**. Clean
white paper on the scanner here gives stripes with the same period on every
sheet, so it is the glass, not the format.

**Finding it:**

```
python paper-sound/spec.py back.wav:song.wav:label
```

It prints where the read has energy the original did not. Real output on sheet
D here:

```
sheetD_scan1 (6606 Hz)
        67.7 Hz   +23.6 dB over the source   (a feature every  97.52 rows)
       108.1 Hz    +2.6 dB over the source   (a feature every  61.13 rows)
      1646.7 Hz   +18.3 dB over the source   (a feature every   4.01 rows)
```

"a feature every N rows" is the pattern's period in sheet rows. Read it like
this:

- **a low, broad peak** (67.7 Hz, ~98 rows) is the slow drift of the page and
  the recording's own bass, not whine; a notch does not cure it;
- **a narrow peak up top** (1646.7 Hz, exactly 4.01 rows) is the glass. A whole
  number of rows per period is the sure sign: the pattern is tied to the
  scanner's rows, not to the sound. On a sheet printed with `--rows 2` the
  frequency is twice that of an ordinary one (here 1647 against 823 Hz: the
  same 8 rows of paper).

On another scanner the frequency will be different; look for the highest
**narrow** peak whose period is close to a whole number of rows.

**Suppressing it**: with a narrow notch filter, outside the reader, when
listening:

```
sox back.wav clean.wav equalizer 823.4 40q -30
sox back.wav clean.wav equalizer 1647  40q -30    # a sheet at --rows 2
```

A high Q (`40q`) is essential: the whine is a pure tone, and the speech around
it has to stay intact.

**Why this is not in the reader itself.** The frequency belongs to a particular
glass, not to the format, so a fixed notch in `read` is impossible.
Subtracting a common per-row pattern was tried and cost more than the tone
itself. This is a matter of listening comfort, not of fixing the read: **the
correlation barely moves with the notch, but it sounds noticeably better**. You
can check the notch did not eat anything useful with the same `score.py`
before and after.

---

## 11. Strips and third-party readers

`cut_lanes.py` cuts a scan into separate strip images, one per lane:

```
python paper-sound/cut_lanes.py sheet.png -o strips
```

A strip is **an image of rows**: a mask around the lane's own stroke, a
negative (ink is light), stretched to a common height. It is material for a
third-party reader (bundled is `picky.py`, a port of Patrick Feaster's Picture
Kymophone) and a diagnostic tool: if `bench` showed scattered bad seconds,
these strips are what to look at.

The cutter prints the most important thing underneath:

```
  one strip is 6608 rows and holds 1/2 of a second, so the audio on it is
  6608 Hz -- and a reader that takes one row for one sample has to play it
  at 13216 Hz
```

**Two different rates, and they must not be confused.** The audio on the strip
is 6608 Hz. But `picky` and everything like it take **a row for a sample** and
know nothing about `--rows 2`, so they have to be given the **row rate**
(13216), otherwise the recording comes out twice as long and an octave lower.

The working command for strips:

```
python paper-sound/picky.py x 50 1 d 165 1 0.95 pdp 13216 24 "strips/*.tif"
```

Two arguments matter here: `165` instead of `330` (no 20 Hz high-pass on each
strip, which costs 0.93 → 0.73) and the row rate instead of the audio rate.
The intensity exponent is left at one: it was measured and means nothing
(LAB §21.2). `picky` now prints what it made of these numbers itself; if the
terminal shows something other than what you expected, the command was typed
wrong.

Two flags beyond the port, both off by default and neither needed on strips
from this cutter:

- `--aperture N`: close the centroid window to N px around the stroke. For
  **someone else's** crop, not aimed by a mask: on a bare cut of
  `sheetD_scan2` it is 0.6314 → 0.9110. On a strip from `cut_lanes` it gives
  nothing.
- `--no-invert`: do not invert a paper-white frame. By default `picky` inverts
  by itself, so the cutter's `--no-negative` no longer breaks it.

**The ceiling of this path** is 0.95 against 0.98 for `paper_sound` on the same
sheet, and that is not a fault: the strips have no clocks (so no retiming) and
no bringing of lanes to a common length in samples. For the sound on a sheet,
go to `paper_sound`.

---

## 12. Frequent questions

**Can I use `--rows 1.5`?** No. Printing would draw it, but the sheet reports
`rows` through the clocks' period, and the reader snaps what it measured onto a
ladder of four steps (20, 32, 40, 64 rows). A fractional value either falls
between the steps (the clocks disappear) or coincides with another one:
20 × 1.6 = 32, so a sheet "at 1.6" is indistinguishable from an ordinary one,
and orientation is lost silently.

**Can I print an "error map" as a second set of curves, to read more
accurately?** Measured: the error repeats between scans of one sheet at
ρ ≈ 0.6…0.77, and a perfect map would give +1.4 dB of median for half the sheet.
Averaging four scans of the sheet gets to +1.0 dB of that (`diversity.py`).
Exactly the same half sheet buys `--rows 2`, which gives +2.1 dB and needs
neither a second pass nor a tie to the scanner.

**Why does the read sound quieter or louder than the original?** It is
normalised by the 99.9th percentile, not by the peak: otherwise one lost lane,
flailing across its full width, would push the whole file down by 11 dB.

**The sheet plays backwards.** Identical clocks (an old sheet): read with
`--upside-down`.

**The second sheet plays at a different pitch.** It was scanned at a different
scale; read both sheets with one command, and the second is brought to the
first.

---

## 13. Symptom → cause

| symptom | cause | what to do |
|---|---|---|
| r below 0.85, no warnings | the centroid window does not match the pair's blur | sweep `--aperture` |
| r near zero, sound unrecognisable | the sheet was segmented at the wrong pitch | `--pitch` by hand; check whether the edge of the glass got into the scan |
| everything after some second is shifted | an extra "lane" in the grid | the `lane N of M` warning; crop the scan to the paper |
| dull sound and hiss | levels tweaked or a white point set while scanning | rescan with no adjustments |
| a thin pure tone over the recording | periodicity of the glass | `spec.py`, then `sox equalizer` (§10) |
| a click once a second | a lane joint; smoothed by itself when read in one command | read all sheets in one call |
| the recording is three or four times as long and lower | a sheet at 3 or 4 read without `--rows` | the `--rows` the reader named (§7) |
| the recording is twice as long and lower | strips played row-per-sample without accounting for `--rows 2` | give the row rate (§11) |
| pale, ragged strokes on the paper | the printer: stroke too thin or too little ink | print darker; larger `--pitch` / `--dpi` per the numbers in the NOTE |
| "the fit has left the sheet ... Rescan it straighter" | the drift fit left the sheet | lay the sheet straighter and rescan |

---

## 14. Where next

- `../README.md`: the format, its measurements and known limits;
- `ARTICLE.md`: what was tried, what failed, and what each step cost in
  decibels;
- `LAB.ru.md`: the lab journal (in Russian): what was measured, what did not
  hold up, what is still open. Every number this guide refers to is there;
- the docstrings in `paper_sound.py` and `read_tracks.py`: why each constant is
  what it is. They are not comments on the code but its justification.
