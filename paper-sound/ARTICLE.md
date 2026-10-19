# Printing sound on paper: what worked, what didn't, and what each step cost

paper-sound prints audio onto an ordinary sheet of paper with an ordinary
office printer, and gets it back with an ordinary flatbed scanner. An A4 sheet
at 600 dpi holds two minutes at 6780 Hz. This is the story of how it got there,
told mostly through the things that didn't work. The full lab journal, 27
entries with every number, is `LAB.ru.md` (in Russian); this is the English
digest.

Every number below uses one metric: the correlation `r` between what was
printed and what came back, expressed as `dB = 10·log10(r² / (1 − r²))`.

## The format in one paragraph

The sheet is cut into vertical lanes. **One lane is one second, one pixel row is
one sample**, and the sample is the horizontal position of a 4 px stroke
wandering inside its lane. The printable height in pixels *is* the sample rate,
so there is no header, no checksum and no metadata: rate, lane pitch, skew and
orientation are all measured off the sheet every time it is read. Two extra
lanes at the edges carry clock tones. That is the whole format.

## Why position, not darkness

Every optical film soundtrack encodes sound in something a single photocell can
measure: brightness (variable density) or the width of a clear area (variable
area). A photocell integrates the whole slit and physically cannot measure
*where* something is.

A 600 dpi scanner can. So the stroke here is a lateral-cut gramophone groove
laid flat, and the reader takes the ink's centroid across the lane. A centroid
is measured against the lane's own centre, so it is blind to how much ink
landed: print density, exposure and the scanner's gamma all drop out.

There is a neat way to compare these carriers. The precision of a position
estimate is bounded by (number of edges) × (how far an edge moves per unit of
signal)². At this lane width a stroke has 2 edges moving 15.9 px each. A
one-sided area track has 1 edge moving 17.9 px, which comes out **2 dB worse**.
A symmetric area track has 2 edges, each getting half the travel, which is
**5 dB worse**. Splitting the swing between edges loses; moving the edges
together wins.

## The first real sheets: the scanner sings

The simulation said one thing; five printed sheets said several others.

The good news first. All five sheets went onto the glass upside down, and the
reader turned all five the right way. The two clocks are printed at different
rates (32 and 20 samples a cycle), so whichever reads back first says which end
is up. That is the one failure in this format that is catastrophic rather than
gradual, and it worked 5 out of 5.

Then every read had a steady high whine: **823.4 Hz**. At this sheet's rate,
that is exactly **8.00 scan rows** per cycle. The scanner has a periodic
pattern in its rows, and since a row is a sample, anything periodic in rows
lands in the audio as a pure tone. Blank paper with no ink at all shows the
same stripe, so it is the glass, not the toner. A sheet printed at two rows per
sample sings at 1647 Hz instead: the same 8 rows of paper.

Nothing in the reader suppresses it. Subtracting a common per-row pattern was
tried, and it also subtracts the part of the sound that every lane shares,
which costs more than the tone. A narrow notch at listening time
(`sox ... equalizer 823.4 40q -30`) barely moves the correlation and makes it
sound much better. The frequency belongs to one particular scanner, so it
cannot be a constant in the reader.

## The test bench lied for three weeks

The scoring script found the best lag between the read and the source by
sliding the source forward. Its own header said, correctly, that print and scan
make a read arrive a sample or two *late*, which needs the opposite slide. So
the code fixed the case that never happens and charged every late read's
misalignment to the paper's noise.

On digital sheets the best lag is exactly 0, which is why this survived. When
it was fixed:

| read | before | after | lag |
|---|---|---|---|
| digital sheet | 32.64 dB | 32.64 dB | 0 |
| sheet B | 6.96 dB | 11.76 dB | +1 |
| sheet C | 1.84 dB | 12.36 dB | +2 |

A "5 dB difference between two formats" was one sample of lag.

The second blow to the bench came a few days later. Two scans of the **same
sheet**, on the same glass, two minutes apart, same settings: **12.98 and
14.93 dB**. And the scan with the *worse* carriage ripple won. So any
difference between two sheets measured with one scan each means nothing below
about 2 dB. Since then, anything that matters is scanned three times and the
medians are compared (`bench.py --same`).

## Paper area: rows beat nothing, width beats rows

Doubling the rows per sample (`--rows 2`) halves how far the stroke has to
travel from one row to the next. On paper it bought **+1.9 dB**, later
confirmed at **+2.1 dB** over four scans, for half the sheet. Halving the lane
width cost **−4.0 dB**. The format sits just above the knee of that curve:
squeezing it is expensive, widening it is cheap but not free.

Doubling the lane width instead of the rows, for the same amount of paper,
was predicted to give +3 dB. On paper it gave +0.6 dB, which is inside the
scan-to-scan spread, so that number counts as not measured at all.

## The top end is saturated, not rolled off

The read loses treble, so the obvious fix is pre-emphasis. A pre-emphasis curve
was fitted to the loss measured off a scan, printed next to the old one that
had been tuned by ear, and lost by **1.24 dB**.

The spectra said why. In the 800–1200 Hz band, a 0.43 dB difference in what was
printed came back as 0.35 dB: linear, everything passes. At 2400–2800 Hz,
**4.31 dB** more ink came back as **0.71 dB**. At 2800–3200 Hz, 6 dB more ink
came back 0.17 dB *quieter*.

A linear filter cannot do that. This is a **slew limit**, not a bandwidth: ink
spread and scanner blur both act *down* the page, which is the time axis. While
the stroke moves less than its own width per row, the blur only smooths it.
Once it moves more, the blur mixes two positions that have nothing to do with
each other. The ear had found the knee before the calculation found the curve:
the hand-tuned "+4 dB at 2600 Hz" sits right on it.

## The scanner settings that silently cost 8 dB

The sub-pixel position lives in the partially covered pixels at the stroke's
edges. A 4 px stroke at x.3 differs from one at x.7 only in how its two edge
pixels share the ink. So:

- **Setting a white point costs 8.6 dB**, silently and irreversibly. Zeroing
  everything paler than 4/255 is enough.
- **Sharpening is worse.** A second flatbed read the same sheets at
  **−19.8 dB** where the first scanner gave −3.95, and at −1.5 dB where the
  first gave +11.9. The sheet itself caught it: a 4 px stroke can only get
  *wider* on its way through toner and glass (six good scans measure 4.32 to
  4.59 px), and this scanner returned **3.2 px**. Something was steepening the
  edges, and the pixels it squared off were the signal.

Both checks now live in the reader. A comb of empty histogram bins means levels
were stretched after the scan. A stroke narrower than printed means something
inside the scanner steepened the edges. The reader can't undo either, but it
tells you to rescan.

## More edges per lane: a good idea that paper refused

The edge-counting rule suggests something tempting. A grating of N bars shifted
as a whole gives 2N edges, all travelling the full swing: +10·log10(N) dB. This
is how optical encoders work.

A test strip with nine gratings beside a reference stroke said the first answer
was **+2.5 dB**, barely a win. But the noise didn't depend on contrast at all: a
block with 15 times less signal was no noisier. The "noise" was scanner
transport jitter shared by every block, including the reference, so the ratio
was dividing one scanner by itself. Measured from the difference between
neighbouring lanes, which cancels the shared part, the grating won by
**+8 dB**, within a decibel of the edge-count prediction.

That was for **still** bars, though. A grating that moves with the audio has to
unwrap its phase, and a vernier simulation put the break-even at 0.65 px of
blur. Two printers and two scanners in all four combinations measured the blur
down the page at **1.10 to 1.44 px**. No available hardware gets anywhere near
0.65, so the idea is closed and the stroke stays.

That same blur turns out to be the format's biggest single loss. A sheet that
never leaves the computer reads at **32.6 dB**; paper reads at **11.8 to
12.3 dB**. About 17 dB of that gap is vertical print and scan blur, confirmed
on all four printer/scanner pairs, and only a different class of printer would
remove it.

## Small wins from outside reviews

Two outside code reviews found real bugs, and the more useful kind: silent ones.

- **One pixel set the ink level for the whole page.** Three functions took the
  darkest pixel as "ink". One speck darker than the print, and a lane was
  declared empty. Taking a percentile of all pixels was the review's suggestion,
  but it just moved the threshold onto the stroke's slope (+7.2 dB on one
  sheet, −2.2 on another). What worked was the 99th percentile of the **per-row
  maxima**: the darkest pixel in each row really is ink, and no single pixel
  gets a vote. One sheet gained 6.6 dB, and none of the other thirteen moved by more than
  0.8.
- **A missing second counted as passed.** A lost lane is replaced by silence,
  its correlation is `NaN`, and `NaN < 0.5` is false. So the "seconds below 0.5"
  line, written for exactly this case, scored a vanished second as fine.
- **A lane lost off the edge of the scan was silent.** A hole in the middle is
  seen from its neighbours, but at the edge there is no other side, so the read
  came out a second short and played as if nothing was wrong. The clocks catch
  it: they are the first and last lanes, so losing an edge lane takes a clock
  with it. That signal had been sitting in the code all along; now it gets
  reported.

## The last experiment: a better estimator that wasn't

After the reviews, the remaining question was clearly how to estimate a 4 px
stroke's position in one row. The candidate was a template fit: build each
lane's average stroke from the sheet itself, then fit every row as
`background + slope·x + gain·template(x − shift)`. This replaces a guessed
gaussian with the real printer+paper+scanner shape.

The first trap appeared on synthetic data. Starting from the plain centroid,
the fit stalled at 0.17 px of error no matter how many passes it made. Starting
from a *randomly noisy* guess, it reached 0.01 px. The centroid's error isn't
random: paper pulls the first moment toward the window's centre, harder the
further the stroke sits from it. So every row with the stroke in one place gets
misaligned the same way, and the template learns that error as the stroke's
shape. You can only build a self-consistent template from a first guess whose
error doesn't depend on position.

With that fixed, it **lost to the plain centroid on every sheet worth
reading**, typically by 0.2 to 0.8 dB. Its one win came on a sheet that reads
below 0 dB even with the centroid, where there was nothing to improve. The template protects against background
and gain wandering from row to row. On these sheets the limit is the noise
inside the stroke itself (ragged toner edges, paper grain), and a model that
says "the stroke has one shape" is wrong about that stroke and adds error of
its own.

## What's left

- **The format pays ~17 dB to vertical blur**, and nothing at hand removes it.
- **The scanner's whine** is handled at listening time, not in the reader.
- **The reader has one live knob, `--aperture`**: the width of the centroid
  window. Too narrow on a clean read costs 18 dB; too wide on a dirty one costs
  0.15 dB. So it stays wide by default.
- **The next sheets** went to a new laser printer and an inkjet, scanned three
  times each on a new scanner. The template fit lost on all twelve scans, worse
  on the inkjet's differently shaped stroke than on the laser's, and was
  deleted. The default aperture was within 0.05 dB of the best on both.

The lesson that kept coming back is that most of the decibels lost here weren't
lost in the format. They were lost in the measurement: a lag searched the wrong
way, a ratio that divided a scanner by itself, a `NaN` that counted as a pass,
one scan treated as the truth. Most of the work went into fixing the bench.

Code, format description and guide: [README](../README.md),
[GUIDE](GUIDE.md).
