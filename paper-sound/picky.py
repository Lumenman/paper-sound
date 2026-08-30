#!/usr/bin/env python3
## Picture Kymophone 1.0 — Python port of picky.m (Windows/Linux)
## Original Copyright (C) 2016 Patrick Feaster
## This program is free software and may be used for any purpose; however, the copyright
## notice must be maintained.  Patrick Feaster is not responsible for any consequences
## of using this software, which is distributed with NO WARRANTY OF ANY KIND.
##
## Requires: numpy, scipy, Pillow   (pip install numpy scipy pillow)
import glob
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat
from scipy.signal import butter, filtfilt

APERTURE = 0    # px of lane the centroid window closes onto, 0 = picky's own
                # first moment of the whole frame.  --aperture
STICKY = 11     # rows the window's aim leans on, 0 or 1 = off.  --sticky
INVERT = True   # a frame whose ground is the bright half is inverted first,
                # so that value and ink are the same thing.  --no-invert

RANGE_FILE = "picky_range.json"  # replaces wdif/sdif/tdif/ndif.mat of the Octave version

# transduction selectors
PDP = {"all", "pos", "dis", "pdp"}
PVL = {"all", "pos", "vel", "pvl"}
IDP = {"all", "int", "dis", "idp"}
IVL = {"all", "int", "vel", "ivl"}
POSITION = {"all", "pdp", "pvl", "pos", "dis", "vel"}
INTENSITY = {"all", "idp", "ivl", "int", "dis", "vel"}

HELP = {
    "h": """\
Command-line argument format: python picky.py w 0 1 d 1 1 0.95 all 44100 24 c:/sourcefile.tif
All arguments are optional but must be entered in order.  Defaults are shown above.  They specify:
w = type of output requested; enter 'picky.py o' for list of options
0 = processing mode; enter 'picky.py p' for list of options
1 = power to which source pixel intensity values will be raised before calculations are performed
d = numerical format, lowercase d or s = double or single precision
    (D and S are accepted for compatibility and behave like d and s)
1 = adjustments of slope and DC offset; enter 'picky.py a' for list of options
1 = sample-to-sample impulse threshold (0=0% of amplitude scale; 1=100% of amplitude scale)
0.95 = decimal value to which amplitude values will be normalized for WAV output
all = type(s) of transduction requested; enter 'picky.py t' for list of options
44100 = sample rate for WAV output
24 = bit depth for WAV output (16, 24, or 32)
c:/sourcefile.tif = file path of source image; batch modes (50+) accept multiple paths here
Glob patterns like strips/strip_*.tif are expanded.  Output files are written
next to the source image(s), not into the current directory.

Three named options may appear anywhere among the arguments above:
--aperture N   close the centroid window down to N px around the stroke and
               read again, instead of taking the first moment of the whole
               frame.  Off by default.  It is for a crop that has NOT been
               masked down to its own stroke already: on one that has it is
               worth nothing, and on one that has not it recovered
               0.6314 -> 0.9110 of median r on a test sheet.
--sticky N     rows the window's aim leans on, default 11.  Has to be chosen
               for the input: 11 suits a masked strip and around 241 an
               unmasked one.  Only meaningful with --aperture.
--no-invert    do not invert a frame whose ground is the bright half.  By
               default a paper-white scan is inverted, because a first moment
               of it follows the paper rather than the ink.""",
    "a": """\
If present, adjustment type must be a number. Options are:
2 = apply 20 Hz high pass filter to single image result
3 = center endpoints of single image result on zero
5 = apply 20 Hz high pass filter to concatenated batch result
7 = center endpoints of concatenated batch result on zero
11 = set first sample from image in batch equal to last sample from previous image
To select multiple options, multiply them.""",
    "o": """\
If present, output type must be a single character.  Options are:
w = Wav, creates WAV file(s) for single images
x = creates concatenated WAV file(s) for a batch, plus reference file with clicks at join points
s = creates concatenated stereo WAV file(s) for a batch with clicks at join points in R channel
v = Vector, saves vector(s) for a single image as .mat
b = Batch, saves concatenated vector(s) for a batch as .mat
m = Matrix, saves matrix/matrices for a batch as .mat
h = Help, prints list of command-line arguments
t = Transduction, prints list of transduction options
o = Output, prints list of output options
p = Processing mode, prints list of processing mode options
a = Adjustments, prints list of adjustment options""",
    "t": """\
If present, transduction type must be a three-character string. Options are:
all = all four available types
pdp = only position displacement (raw positional values)
pvl = only position velocity (derivative/rate of change of positional values)
idp = only intensity displacement (raw intensity values)
ivl = only intensity velocity (derivative/rate of change of intensity values)
pos = both position-based options (pdp and pvl)
int = both intensity-based options (idp and ivl)
dis = both displacement-based options (pdp and idp)
vel = both velocity-based options (pvl and ivl)
Any other three-character string will suppress file generation.""",
    "p": """\
If present, processing mode type must be a number. Options are:
0 = Process single image independently
1 = Process single image independently; save amplitude range values
2 = Process single image independently; overwrite any saved amplitude range values that are exceeded
3 = Process single image using previously saved amplitude range values
50 = Process multiple images independently from one another
51 = Process multiple images to consistent amplitude range auto-detected for group
52 = Process multiple images using previously saved amplitude range values""",
}


def die(msg):
    print(msg)
    sys.exit(1)


def write_wav(filename, data, samplerate, bitdepth):
    """Write float data in [-1, 1] as a PCM WAV; mono (n,) or multi-channel (n, ch)."""
    data = np.clip(np.asarray(data, np.float64), -1.0, 1.0)
    if data.ndim == 1:
        data = data[:, None]
    peak = 2 ** (bitdepth - 1) - 1
    ints = np.round(data * peak).astype("<i4")
    if bitdepth == 16:
        raw = ints.astype("<i2").tobytes()
    elif bitdepth == 24:
        raw = ints.reshape(-1, 1).view(np.uint8)[:, :3].tobytes()
    elif bitdepth == 32:
        raw = ints.tobytes()
    else:
        die(f"{bitdepth}: Unsupported bit depth (use 16, 24, or 32).")
    with wave.open(str(filename), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(bitdepth // 8)
        w.setframerate(int(samplerate))
        w.writeframes(raw)


def reject_impulses(D, Pc, saved_dif):
    """Impulse noise attenuation on derivative vector D, in place (mirrors picky.m)."""
    if len(D) == 0:
        return
    maxcl = max(abs(float(D.min())), float(D.max()))  # fixed at initial level throughout
    if saved_dif is not None and saved_dif / 2 > maxcl:
        maxcl = saved_dif / 2
    thr = Pc * maxcl
    for i in range(len(D)):
        if abs(D[i]) > thr:
            if i == 0:
                D[i] = 0
            else:
                repl = None
                for j in range(1, 11):  # search up to 10 samples ahead
                    if i + j >= len(D):
                        break
                    if abs(D[i + j]) < thr:
                        repl = (D[i - 1] + D[i + j]) / 2
                        break
                D[i] = repl if repl is not None else D[i - 1]


def center_endpoints(V):
    n = len(V)
    start = -V[0]
    stop = -V[-1] - start
    return V + start + stop * np.arange(1, n + 1) / n


def highpass20(V, samplerate):
    # filtfilt rather than lfilter, which is what picky.m had: a second-order
    # causal filter puts a transient at the start of whatever it is handed and
    # a phase slope across the band, and both land in the wav. Zero phase
    # costs one word. Measured on its own, median r over the seconds against
    # the audio that went onto the paper: sheetC's strips 0.8492 -> 0.8577,
    # 01's 0.8658 -> 0.8703, sheetD_scan2's 0.8901 -> 0.8878. Two up, one
    # marginally down, and a second comes out of the bad tail on the first
    # two.
    if len(V) <= 15:
        return V              # shorter than the filter's own settling
    b, a = butter(2, 20 / (samplerate / 2), "high")
    return filtfilt(b, a, V)


def box1d(a, n):
    """Moving average over n samples, edges held."""
    if n < 2:
        return a
    pad = np.pad(a, (n // 2, n - 1 - n // 2), mode="edge")
    c = np.pad(np.cumsum(pad), (1, 0))
    return (c[n:] - c[:-n]) / n


def narrow_centroid(F, W, narrow, sticky, passes=12):
    """Read W again with the window closed onto the stroke it just found.

    picky's own W is the first moment of the whole frame. On a strip that is
    nine tenths bare paper the paper outvotes the stroke, and on a crop cut to
    a lane the neighbouring lane's stroke swings into it. Closing the window
    onto the stroke and reading again is what paper_sound's reader does, and
    the aim leans on the neighbouring rows -- the stroke moves a fifth of a
    pixel a row, so what a dozen rows agree on is where this row's stroke is,
    and a row whose window fell on paper is outvoted instead of followed.
    Only the AIM leans; the sample is always the raw centroid of the pass.

    Off by default, because on a strip that has already been masked down to
    its own stroke there is nothing left to reject and the pass is worth
    nothing. It is for a crop that has NOT been masked. Measured on an
    unmasked cut of sheetD_scan2 (41 px pitch), median r over the seconds:

        picky as it stands              0.6314
        --aperture 8 --sticky 0         0.5115   window loses the stroke
        --aperture 8 --sticky 11        0.7226
        --aperture 8 --sticky 241       0.9110
        the same sheet cut WITH a mask  0.9548

    So it recovers most of what the mask is worth and not all of it: a strip
    does not carry where the neighbouring lanes are, and on a 41 px pitch that
    is the last 3 dB. On a wide pitch it recovers the lot -- sheetC, 79 px
    lanes, reads 0.8553 unmasked with --aperture 16 against 0.8577 masked
    without it. On a masked strip it is worth nothing at all and can cost:
    sheetD_scan2 masked reads 0.9548 plain and 0.9536 with --aperture 8.

    `sticky` is the one number that has to be chosen for the input rather than
    once: 11 on a masked strip, ~241 on an unmasked one, and 241 on a masked
    one costs 0.9536 -> 0.9435. It scales with how many rows the paper spends
    per sample, which the picture does not say.
    """
    F = np.asarray(F, np.float64)
    h = F.shape[0]
    x = np.arange(h, dtype=np.float64)
    cs = np.pad(np.cumsum(F, axis=0), ((1, 0), (0, 0)))
    cx = np.pad(np.cumsum(F * x[:, None], axis=0), ((1, 0), (0, 0)))
    cols = np.arange(F.shape[1])

    def at(c, p):
        """Cumulative sum at a fractional row, per column."""
        p = np.clip(p, 0, h)
        i = np.floor(p).astype(int)
        j = np.minimum(i + 1, h)
        f = p - i
        return c[i, cols] * (1 - f) + c[j, cols] * f

    guess = h - W          # picky counts rows up from the bottom; this counts down
    # Flat at four apertures for the sticky passes, then down onto the
    # stroke, which is paper_sound's own schedule. A taper from the whole
    # frame down was tried instead, on the reasoning that picky is not told
    # how wide a lane is: it reads 0.9000 against 0.9110 on an unmasked cut of
    # sheetD_scan2, because the wider a mis-aimed window is the more of the
    # neighbouring lane it collects to be mis-aimed by. Kept flat.
    wide = min(4 * narrow, h)
    schedule = [(wide, sticky)] * passes + [(2 * narrow, 0), (narrow, 0)]
    for width, lean in schedule:
        aim = box1d(guess, lean) if lean > 1 else guess
        lo = aim - width / 2
        den = at(cs, lo + width) - at(cs, lo)
        num = at(cx, lo + width) - at(cx, lo)
        ok = den > 0.05 * np.median(den)          # window that fell on paper
        fine = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
        guess = np.where(np.isfinite(fine), fine, aim)
    return h - guess


def scale_vec(V, wavscale, vmin=None, vmax=None):
    """Scale to -wavscale..+wavscale (or zeros if flat)."""
    if vmin is None:
        vmin = float(V.min())
    if vmax is None:
        vmax = float(V.max())
    absmax = abs(vmin) if abs(vmin) > vmax else vmax
    return np.zeros_like(V) if absmax == 0 else V / absmax * wavscale


def load_ranges(outdir):
    try:
        with open(outdir / RANGE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        die(f"Saved amplitude range values not found ({outdir / RANGE_FILE}); "
            "run processing mode 1 first.")


def save_ranges(outdir, r):
    with open(outdir / RANGE_FILE, "w") as f:
        json.dump(r, f)


def process_single(path, output, Emode, Ipower, Fmat, corr1, Pc, wavscale,
                   transduction, samplerate, bitdepth, statedir=None):
    name = Path(path).name
    # outputs live next to the source image, not in the cwd, which for a
    # double-clicked run can be some unrelated directory
    outdir = Path(path).resolve().parent
    # a batch anchors its shared range file to one directory, so modes 51/52
    # still work when the images are spread across several folders
    statedir = outdir if statedir is None else statedir
    try:
        F = np.asarray(Image.open(path))
    except Exception as e:
        die(f"{name}: cannot read image ({e}).")
    if F.ndim > 3:
        die("Source image has too many dimensions.")
    if F.ndim == 3:
        if F.shape[2] in (2, 4):
            F = F[..., :-1]  # alpha is a separate output of Octave's imread, not a channel
        F = F.sum(axis=2)
        print(f"Multi-channel source image ({name}) summed to single channel")
    if F.shape[0] > F.shape[1]:  # taller than wide: rotate 90 degrees counterclockwise
        F = np.rot90(F)

    # Ink dark on paper-white is the ordinary way a curve is scanned, and a
    # first moment of THAT follows the paper: the trace is a thin dip in a
    # bright field, so what comes back is diluted and mirrored -- r = -1.000
    # against the audio, on a sheet printed from it. Measured on an unmasked
    # cut of sheetD_scan2, median r over the seconds: 0.3841 paper-white
    # against 0.6314 inverted, and with --aperture on, 0.1567 against 0.9110,
    # because a window told to close on the brightest thing in a paper-white
    # frame closes on paper. Judged by the median: a curve fills a tenth of
    # its frame, so the median is the ground, and a ground above the middle of
    # the frame's own range is the light one. --no-invert turns it off.
    lo, hi = float(F.min()), float(F.max())
    if INVERT and hi > lo and float(np.median(F)) > (lo + hi) / 2:
        F = hi - F
        print(f"{name}: ink is dark on a light ground; inverted")

    dtype = np.float32 if Fmat in ("s", "S") else np.float64
    F = F.astype(dtype) ** Ipower
    S = F.sum(axis=0)  # column intensity sums
    with np.errstate(invalid="ignore", divide="ignore"):
        Q = F / S  # pixel intensities as fractions of total column intensity
    Q[np.isnan(Q)] = 0
    Z = np.arange(F.shape[0], 0, -1, dtype=dtype)[:, None]  # descending row numbers
    W = (Z * Q).sum(axis=0)  # intensity-weighted row position per column
    # A column with no ink in it at all is not a reading of zero, it is no
    # reading. picky.m let the NaN fall through to W = 0, and after the median
    # subtraction below that is a spike of half the frame's height -- a click,
    # not a sample. They are not rare on a masked strip: a cut of sheetD_scan2
    # has 1137 ink-free columns in 792960, and holding the trace across them
    # instead of dropping it to the floor is worth 0.8901 -> 0.9543 of median
    # r against the audio that was printed, which is more than everything else
    # in this file put together. S is left alone: no ink really is no
    # intensity, so the idp/ivl readings of such a column are honest zeros.
    lit = S > 0
    if lit.any() and not lit.all():
        W = np.interp(np.arange(len(W)), np.flatnonzero(lit), W[lit])
    if APERTURE:
        W = narrow_centroid(F, W, APERTURE, STICKY)
    T = np.diff(W)  # position velocity
    N = np.diff(S)  # intensity velocity

    ranges = load_ranges(statedir) if Emode in (2, 3) else None
    if Pc < 1 and transduction in POSITION:
        reject_impulses(T, Pc, ranges["Tdif"] if Emode == 3 else None)
        W = np.concatenate(([W[0]], W[0] + np.cumsum(T)))  # re-integrate W
    if Pc < 1 and transduction in INTENSITY:
        reject_impulses(N, Pc, ranges["Ndif"] if Emode == 3 else None)
        S = np.concatenate(([S[0]], S[0] + np.cumsum(N)))  # re-integrate S

    W = W - np.median(W)
    S = S - np.median(S)
    T = T - np.median(T)
    N = N - np.median(N)

    if corr1 % 3 == 0:
        # fixes a copy-paste typo in picky.m, which corrected T twice and never N
        # (the batch-level corr 7 block shows the intended W/T/S/N pattern)
        W = center_endpoints(W)
        T = center_endpoints(T)
        S = center_endpoints(S)
        N = center_endpoints(N)
    if corr1 % 2 == 0:
        W = highpass20(W, samplerate)
        S = highpass20(S, samplerate)

    lims = {k: [float(V.min()), float(V.max())] for k, V in
            (("W", W), ("S", S), ("T", T), ("N", N))}
    if Emode == 1:  # save current range values
        save_ranges(statedir, {k + "dif": hi - lo for k, (lo, hi) in lims.items()})
    elif Emode == 2:  # widen saved range values if exceeded
        for k, (lo, hi) in lims.items():
            if hi - lo > ranges[k + "dif"]:
                ranges[k + "dif"] = hi - lo
        save_ranges(statedir, ranges)
    elif Emode == 3:  # expand to saved range, excess spread evenly top and bottom
        for k, (lo, hi) in lims.items():
            dif = ranges[k + "dif"]
            if dif > hi - lo:
                margin = (dif - (hi - lo)) / 2
                lims[k] = [lo - margin, hi + margin]

    timestring = str(int(time.time()))
    stem = Path(name).stem
    for key, V, sel, tag in (("W", W, PDP, "pdp"), ("T", T, PVL, "pvl"),
                             ("S", S, IDP, "idp"), ("N", N, IVL, "ivl")):
        if transduction not in sel:
            continue
        if output == "w":
            lo, hi = lims[key]
            write_wav(outdir / f"{stem}_{timestring}_{tag}.wav",
                      scale_vec(V, wavscale, lo, hi), samplerate, bitdepth)
        elif output == "v":
            savemat(str(outdir / (key.lower() + ".mat")), {key: V})

    print(f"{name} processed mode_{Emode}_{output}_{transduction} ID={timestring}"
          f"  ({len(W)} columns at {samplerate:g} Hz = {len(W) / samplerate:.2f} s)")
    if W.max() - W.min() == 0:
        print(f"Output value range for ({name}) is zero; "
              "recommend retry with different numerical format setting")
    return {"W": W, "T": T, "S": S, "N": N}


def process_batch(files, output, Emode, Ipower, Fmat, corr1, Pc, wavscale,
                  transduction, samplerate, bitdepth):
    outdir = Path(files[0]).resolve().parent  # batch outputs go next to the first image
    if Emode == 51:  # preliminary pass to auto-detect consistent amplitude range
        process_single(files[0], "-", 1, Ipower, Fmat, corr1, Pc, wavscale,
                       "000", samplerate, bitdepth, outdir)
        for f in files[1:]:
            process_single(f, "-", 2, Ipower, Fmat, corr1, Pc, wavscale,
                           "000", samplerate, bitdepth, outdir)
        print("Amplitude levels set for batch")

    per_mode = 3 if Emode in (51, 52) else 0
    concat = output in ("x", "b", "s")

    if concat or output == "m":
        conc = {k: [np.zeros(3000)] for k in ("W", "S", "T", "N", "click")}  # leading silence
        rows = {k: [] for k in ("W", "S", "T", "N")}
        last = None
        for f in files:
            r = process_single(f, "-", per_mode, Ipower, Fmat, corr1, Pc, wavscale,
                               transduction, samplerate, bitdepth, outdir)
            W, S, T, N = r["W"], r["S"], r["T"], r["N"]
            if corr1 % 11 == 0 and last is not None:
                # set first sample equal to last sample of previous image
                W = W + (last["W"] - W[0])
                S = S + (last["S"] - S[0])
                T = T + (last["T"] - T[0])
                N = N + (last["N"] - N[0])
            for k, V in (("W", W), ("S", S), ("T", T), ("N", N)):
                rows[k].append(V)
            if concat:
                conc["W"].append(W)
                conc["S"].append(S)
                # first image: double first velocity sample to stay in sync with clickfile;
                # later images: transitional velocity sample between the pair
                tjoin = T[0] if last is None else last["W"] - W[0]
                njoin = N[0] if last is None else last["S"] - S[0]
                conc["T"].append(np.concatenate(([tjoin], T)))
                conc["N"].append(np.concatenate(([njoin], N)))
                ck = np.zeros(len(W))
                ck[0], ck[-1] = 1, -1  # clicks at image joins
                conc["click"].append(ck)
                print("Result concatenated")
            last = {"W": W[-1], "S": S[-1], "T": T[-1], "N": N[-1]}
    else:  # handle images separately (e.g. output w)
        for f in files:
            process_single(f, output, per_mode, Ipower, Fmat, corr1, Pc, wavscale,
                           transduction, samplerate, bitdepth, outdir)

    timestring = str(int(time.time()))

    if output == "m":
        for key, matname, varname, sel in (("W", "wmatrix", "Wmx", PDP),
                                           ("T", "tmatrix", "Tmx", PVL),
                                           ("S", "smatrix", "Smx", IDP),
                                           ("N", "nmatrix", "Nmx", IVL)):
            if transduction not in sel:
                continue
            L = max(len(v) for v in rows[key])
            M = np.zeros((len(files), L))
            for i, v in enumerate(rows[key]):
                M[i, :len(v)] = v
            savemat(str(outdir / (matname + ".mat")), {varname: M})

    if concat:
        for k in conc:
            conc[k].append(np.zeros(3000))  # trailing silence
        C = {k: np.concatenate(v) for k, v in conc.items()}
        if corr1 % 7 == 0:
            for k in ("W", "T", "S", "N"):
                C[k] = center_endpoints(C[k])
        if corr1 % 5 == 0:
            for k in ("W", "S", "T", "N"):
                V = highpass20(C[k] - 0.5 * C[k].max(), samplerate)
                V[:3000] = 0  # reset initial and terminal silences to zero
                V[-3000:] = 0
                C[k] = V
        click = C["click"]
        if output == "x":
            write_wav(outdir / f"Batch_{timestring}_clk.wav", click, samplerate, bitdepth)
        if output == "b":
            savemat(str(outdir / "clk.mat"), {"clickfile": click})
        for key, tag, sel in (("W", "pdp", PDP), ("T", "pvl", PVL),
                              ("S", "idp", IDP), ("N", "ivl", IVL)):
            if transduction not in sel:
                continue
            sc = scale_vec(C[key], wavscale)
            if output == "x":
                write_wav(outdir / f"Batch_{timestring}_{tag}.wav", sc, samplerate, bitdepth)
            if output == "s":  # stereo: signal left, clicks right
                write_wav(outdir / f"Batch_{timestring}_{tag}st.wav",
                          np.column_stack([sc, click]), samplerate, bitdepth)
            if output == "b":
                savemat(str(outdir / (key.lower() + "conc.mat")), {key + "conc": C[key]})

    if output in ("x", "b", "m", "s"):
        # `m` writes a matrix, one row per image, so there is no total to give
        total = f"  ({len(C['W']) / samplerate:.2f} s total)" if concat else ""
        print(f"Concatenated batch results exported mode_{Emode}_{output}_{transduction} "
              f"ID={timestring}{total}")
    print("Requested batch processing complete")


def pick_files(multiple):
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
    except Exception:
        die("No file dialog available; pass image path(s) on the command line.")
    root.withdraw()
    ft = [("Image files", "*.png *.gif *.bmp *.tif *.tiff *.jpg *.jpeg *.webp"),
          ("All files", "*.*")]
    if multiple:
        sel = list(filedialog.askopenfilenames(title="Select MULTIPLE image files", filetypes=ft))
    else:
        one = filedialog.askopenfilename(title="Select an image file", filetypes=ft)
        sel = [one] if one else []
    root.destroy()
    return sel


def take_opt(a, name, default):
    """Pull `--name VALUE` out of the argument list, wherever it sits.

    The ten positional arguments are picky.m's own and are matched by
    position, so anything added has to be taken out of the list before they
    are counted -- and named, because an eleventh position is where the file
    paths start.
    """
    if name not in a:
        return default
    i = a.index(name)
    if i + 1 >= len(a):
        die(f"{name}: needs a value.")
    v = to_num(a[i + 1], name)
    del a[i:i + 2]
    return v


def take_flag(a, name):
    if name in a:
        a.remove(name)
        return True
    return False


def to_num(s, what):
    try:
        return float(s)
    except (TypeError, ValueError):
        die(f"{s}: Invalid value for {what}.")


def draw(h, n, centre, stroke=4.0, value=255.0):
    """A strip with one stroke of `stroke` px running along `centre`."""
    y = np.arange(h, dtype=np.float64)[:, None]
    return value * np.clip(stroke / 2 + 0.5 - np.abs(y - centre[None, :]), 0, 1)


def selftest():
    """Draw a stroke, read it back, and check each piece earns its place."""
    rng = np.random.default_rng(7)
    h, n = 42, 4000            # one lane and its bleed, at the printed pitch
    t = np.arange(n)
    # Periods in COLUMNS, chosen the way paper counts: a real stroke moves a
    # fifth of a pixel per row, and a synthetic one that moves five would be
    # asking the sticky aim to follow something no scan contains.
    centre = h / 2 + 10.4 * np.sin(2 * np.pi * t / 1500) + 2.6 * np.sin(2 * np.pi * t / 260)

    def r(read):
        return float(np.corrcoef(read - read.mean(), centre - centre.mean())[0, 1])

    def readback(F):
        """picky's own transduction, straight through, with no file involved."""
        F = np.asarray(F, np.float64)
        S = F.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Q = F / S
        Q[np.isnan(Q)] = 0
        W = (np.arange(F.shape[0], 0, -1, dtype=np.float64)[:, None] * Q).sum(axis=0)
        lit = S > 0
        if lit.any() and not lit.all():
            W = np.interp(np.arange(len(W)), np.flatnonzero(lit), W[lit])
        return W, S

    # box1d holds the ends and averages the middle
    a = np.arange(10, dtype=np.float64)
    assert np.allclose(box1d(a, 3)[1:-1], a[1:-1]), "box1d bends a straight line"
    assert len(box1d(a, 4)) == len(a), "box1d changed the length"
    assert box1d(a, 1) is a, "box1d should be a no-op below 2"

    # a clean strip reads back whatever the window, so the aperture must not
    # break what already worked
    clean = draw(h, n, centre)
    W, _ = readback(clean)
    assert r(-W) > 0.999, f"clean strip reads {r(-W):.4f}"
    assert r(-narrow_centroid(clean, W, 8, 11)) > 0.999, "aperture broke a clean read"

    # ink-free columns are held across, not dropped to the floor
    holed = clean.copy()
    holed[:, ::500] = 0
    W, _ = readback(holed)
    assert r(-W) > 0.999, f"ink-free columns spike: {r(-W):.4f}"
    assert abs(W[500] - W[499]) < 1.0, "an ink-free column read as a step"

    # a dirty wide lane is what the aperture is for. Grain over the whole
    # frame, at the level paper_sound measured its own window on: a stroke
    # fills a tenth of its lane, so nine tenths of what a frame-wide first
    # moment weighs is paper, and the further out that paper sits the more
    # leverage it has on the moment.
    dirty = clean + rng.uniform(0, 60, (h, n))
    W, _ = readback(dirty)
    wide = r(-W)
    narrow = r(-narrow_centroid(dirty, W, 8, 11))
    assert narrow > 0.999 > wide, f"aperture {narrow:.4f} against wide {wide:.4f}"

    # a paper-white frame is inverted, and reads the same as its negative
    white = 255.0 - clean
    assert np.median(white) > (white.min() + white.max()) / 2, "invert would not fire"
    Wi, _ = readback(white.max() - white)
    assert r(-Wi) > 0.999, "inverted frame does not read back"

    # filtfilt leaves a straight line straight, where lfilter would not
    v = highpass20(np.ones(4000), 6579)
    assert abs(v[100:-100]).max() < 1e-6, "high pass rings in the middle of DC"

    print(f"selftest ok -- a grainy lane reads {wide:.4f} frame-wide and "
          f"{narrow:.4f} at --aperture 8")


def main(argv=None):
    global APERTURE, STICKY, INVERT
    a = list(sys.argv[1:] if argv is None else argv)
    if take_flag(a, "--selftest"):
        return selftest()
    APERTURE = take_opt(a, "--aperture", APERTURE)
    STICKY = int(take_opt(a, "--sticky", STICKY))
    INVERT = not take_flag(a, "--no-invert")
    defaults = ["w", "0", "1", "d", "1", "1", "0.95", "all", "44100", "24"]
    vals = a[:10] + defaults[len(a):]
    # expand glob patterns here: cmd.exe and PowerShell pass them through as-is
    files = [f for x in a[10:] for f in (sorted(glob.glob(x)) or [x])]
    (output, Emode, Ipower, Fmat, corr1, Pc,
     wavscale, transduction, samplerate, bitdepth) = vals

    if len(transduction) != 3:
        die(f"{transduction}: Invalid value for transduction type(s).")
    if len(output) != 1:
        die(f"{output}: Invalid value for output type(s).")
    if Fmat not in ("d", "D", "s", "S"):
        die(f"{Fmat}: Invalid value for numerical format configuration.")
    if output in HELP:
        print(HELP[output])
        return

    Emode = int(to_num(Emode, "processing mode"))
    Ipower = to_num(Ipower, "power to which pixel intensity values will be raised")
    corr1 = int(to_num(corr1, "slope and DC offset adjustment"))
    Pc = to_num(Pc, "impulse rejection threshold")
    wavscale = to_num(wavscale, "amplitude normalization")
    samplerate = to_num(samplerate, "sample rate")
    bitdepth = int(to_num(bitdepth, "bit depth"))
    if corr1 < 1:
        die(f"{corr1}: Invalid value for slope and DC offset adjustment.")

    # corr1 is a PRODUCT of flags, so 330 and 165 are one digit apart and a
    # whole 20 Hz filter apart, and the wav says nothing about which was run.
    # Two days of a sheet reading as noise came down to that digit, and to a
    # sample rate given in samples where the strips were counted in rows.
    # Both are printed here, where they can still be read off the terminal.
    on = [t for k, t in ((2, "20 Hz high pass per image"),
                         (3, "endpoints centred per image"),
                         (5, "20 Hz high pass on the batch"),
                         (7, "endpoints centred on the batch"),
                         (11, "joins made continuous"))
          if corr1 % k == 0]
    print(f"adjustments {corr1}: {', '.join(on) if on else 'none'}")
    if samplerate <= 0:
        die(f"{samplerate}: Invalid value for sample rate.")
    if bitdepth <= 0:
        die(f"{bitdepth}: Invalid value for bit depth.")

    if Emode < 50:
        if files:
            src = files[0]
        else:
            picked = pick_files(multiple=False)
            if not picked:
                die("No source image selected.")
            src = picked[0]
        process_single(src, output, Emode, Ipower, Fmat, corr1, Pc, wavscale,
                       transduction, samplerate, bitdepth)
    else:
        if not files:
            files = pick_files(multiple=True)
        if len(files) < 2:
            die(f"Requested processing mode ({Emode}) requires multiple image selection.")
        process_batch(files, output, Emode, Ipower, Fmat, corr1, Pc, wavscale,
                      transduction, samplerate, bitdepth)


if __name__ == "__main__":
    main()
