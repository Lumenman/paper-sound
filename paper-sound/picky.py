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
from scipy.signal import butter, lfilter

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
next to the source image(s), not into the current directory.""",
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
    b, a = butter(2, 20 / (samplerate / 2), "high")
    return lfilter(b, a, V)


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

    dtype = np.float32 if Fmat in ("s", "S") else np.float64
    F = F.astype(dtype) ** Ipower
    S = F.sum(axis=0)  # column intensity sums
    with np.errstate(invalid="ignore", divide="ignore"):
        Q = F / S  # pixel intensities as fractions of total column intensity
    Q[np.isnan(Q)] = 0
    Z = np.arange(F.shape[0], 0, -1, dtype=dtype)[:, None]  # descending row numbers
    W = (Z * Q).sum(axis=0)  # intensity-weighted row position per column
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

    print(f"{name} processed mode_{Emode}_{output}_{transduction} ID={timestring}")
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
        print(f"Concatenated batch results exported mode_{Emode}_{output}_{transduction} "
              f"ID={timestring}")
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


def to_num(s, what):
    try:
        return float(s)
    except (TypeError, ValueError):
        die(f"{s}: Invalid value for {what}.")


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
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
