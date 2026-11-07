"""Plot model-talker mid-vowel F1/F2 (LADAL-style: reversed axes, tokens, 1.5 sd ellipses, means).

python plots/mid_vowel_chart.py [--final-only]  (one panel per recording version, v1 and v2)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from matplotlib.patches import Ellipse
from parselmouth.praat import call

ROOT = Path(__file__).resolve().parent.parent
VOWELS = "eɛoɔøœ"
COLOR = {"e": "#1f77b4", "ɛ": "#d62728", "o": "#2ca02c", "ɔ": "#ff7f0e", "ø": "#9467bd", "œ": "#8c564b"}
NUCLEI = set("aeiouyɛɔøœəɑɐɪʊ")
MIN_DUR = 0.06
CEILINGS = (5000, 5500, 6000)


def tokens(audio_dir, final_only):
    """tokens returns (vowel, F1, F2, sentence, version) for each mid vowel of at least 60 ms."""
    out = []
    for wav in sorted(audio_dir.glob("*.wav")):
        sentence, _, version = wav.stem.partition("_")
        snd = parselmouth.Sound(str(wav))
        tg = call("Read from file", str(wav.with_suffix(".TextGrid")))
        fs = [snd.to_formant_burg(max_number_of_formants=5, maximum_formant=m) for m in CEILINGS]
        n = call(tg, "Get number of intervals", 2)
        for j in range(1, n + 1):
            lab = call(tg, "Get label of interval", 2, j)
            x, y = (call(tg, f"Get {p} time of interval", 2, j) for p in ("start", "end"))
            if lab not in set(VOWELS) or y - x < MIN_DUR:
                continue
            if final_only:
                word = call(tg, "Get interval at time", 1, (x + y) / 2)
                w_end = call(tg, "Get end time of interval", 1, word)
                later = (
                    call(tg, "Get label of interval", 2, k)
                    for k in range(j + 1, n + 1)
                    if call(tg, "Get start time of interval", 2, k) < w_end - 1e-4
                )
                if any(lbl[:1] in NUCLEI for lbl in later):
                    continue
            ts = np.linspace(x + (y - x) / 4, y - (y - x) / 4, 7)
            f1, f2 = (np.nanmedian([f.get_value_at_time(k, t) for f in fs for t in ts]) for k in (1, 2))
            if np.isfinite(f1) and np.isfinite(f2):
                out.append((lab, f1, f2, sentence, version))
    return out


def ellipse(ax, pts, color):
    """ellipse draws a 1.5 sd covariance ellipse, skipped under 4 points."""
    if len(pts) < 4:
        return
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    ang = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    w, h = 2 * 1.5 * np.sqrt(vals[::-1])
    ax.add_patch(Ellipse(pts.mean(0), w, h, angle=ang, fill=False, ec=color, lw=1.5, alpha=0.8))


def panel(ax, toks, title):
    for lab in VOWELS:
        pts = np.array([[t[2], t[1]] for t in toks if t[0] == lab])
        if not len(pts):
            continue
        for x, y in pts:
            ax.text(x, y, lab, color=COLOR[lab], ha="center", va="center", fontsize=11, alpha=0.7)
        ellipse(ax, pts, COLOR[lab])
        m = pts.mean(0)
        ax.text(
            *m, lab, color="white", ha="center", va="center", fontsize=16, fontweight="bold",
            bbox={"boxstyle": "circle", "fc": COLOR[lab], "ec": "none"},
        )  # fmt: skip
    ax.set_title(f"{title} (n={len(toks)})")
    ax.set_xlabel("F2 (Hz)")
    ax.grid(alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-only", action="store_true", help="drop vowels that are not in the word's last syllable")
    ap.add_argument("--audio", type=Path, default=ROOT / "corpus" / "model_denoised")
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()

    toks = tokens(args.audio, args.final_only)
    groups = [(v, [t for t in toks if t[4] == v]) for v in ("v1", "v2")]
    stem = "mid_vowels"
    if args.final_only:
        stem += "_final"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (title, g) in zip(axes, groups):
        panel(ax, g, title)
    axes[0].set_ylabel("F1 (Hz)")
    axes[0].set_xlim(2500, 400)
    axes[0].set_ylim(750, 250)
    scope = "Word-final mid vowels" if args.final_only else "Mid vowels"
    fig.suptitle(f"{scope}, model talker (tokens >= 60 ms; ellipses 1.5 sd, need 4+ tokens; filled = mean)")
    fig.tight_layout()
    out = args.out or ROOT / "plots" / f"{stem}.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
