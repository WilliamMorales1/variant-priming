"""Bigger synthetic corpus. Every recording (model talker and participant) is synthesized twice:
continuously (the real file) and chunk by chunk (known site times, mapped later)."""

import json
import sys
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

S = Path(sys.argv[1])
corpus = Path(sys.argv[2])
siwis = PiperVoice.load(str(S / "voices/fr_FR-siwis-medium.onnx"))
upmc = PiperVoice.load(str(S / "voices/fr_FR-upmc-medium.onnx"))

# (site, phonemes A, phonemes B); site values use SAMPA-ish codes.
SENT = {
    "S01": [
        (None, "le ʒœn", "le ʒœn"),
        ("V1", "pɛ", "pe"),
        (None, "ʃœʁ ɔ̃ at", "ʃœʁ ɔ̃ at"),
        ("R1", "ʁa", "ʁa"),
        (None, "pe dø", "pe dø"),
        ("R2", "pwa", "pwa"),
        (None, "sɔ̃", "sɔ̃"),
    ],
    "S02": [
        (None, "mez", "mez"),
        ("R1", "ami", "ami"),
        (None, "ɑ̃ɡlɛ sɔ̃ ale a", "ɑ̃ɡlɛ sɔ̃ ale a"),
        ("R2", "la", "la"),
        ("V1", "mɛ", "me"),
        (None, "zɔ̃", "zɔ̃"),
    ],
    "S03": [
        (None, "lə", "lə"),
        ("R1", "ɡaʁ", "ɡaʁ"),
        (None, "sɔ̃", "sɔ̃"),
        ("V1", "ɛ", "e"),
        (None, "mɛ bjɛ̃ lə", "mɛ bjɛ̃ lə"),
        ("V2", "ʁɔ", "ʁo"),
        (None, "zje dy", "zje dy"),
        ("R2", "ʒaʁ", "ʒaʁ"),
        (None, "dɛ̃", "dɛ̃"),
    ],
    "S04": [
        ("R1", "nu", "nu"),
        (None, "dəvɔ̃", "dəvɔ̃"),
        ("V1", "ʒø", "ʒœ"),
        (None, "ne avɑ̃ lə", "ne avɑ̃ lə"),
        ("V2", "bo", "bɔ"),
        ("R2", "ku", "ku"),
        (None, "də", "də"),
        ("R3", "vi", "vi"),
        (None, "zit", "zit"),
    ],
    "S05": [
        (None, "ɛl", "ɛl"),
        ("R1", "a", "a"),
        ("V1", "e", "ɛ"),
        (None, "te ɑ̃ fʁɑ̃s pɑ̃dɑ̃ l", "te ɑ̃ fʁɑ̃s pɑ̃dɑ̃ l"),
        ("V2", "o", "ɔ"),
        (None, "tɔn a pa", "tɔn a pa"),
        ("R2", "ʁi", "ʁi"),
    ],
    "S06": [
        (None, "lə", "lə"),
        ("V1", "pœ", "pø"),
        (None, "plije ɛ ʁə", "plije ɛ ʁə"),
        ("V2", "pɔ", "po"),
        (None, "ze pʁɛ də", "ze pʁɛ də"),
        ("R1", "la", "la"),
        ("R2", "ʁi", "ʁi"),
        (None, "vjɛʁ", "vjɛʁ"),
    ],
}
CODE = {"ɛ": "E", "e": "e", "ɔ": "O", "o": "o", "ø": "2", "œ": "9", "a": "a", "i": "i", "u": "u"}


def value(ph):
    return next(CODE[c] for c in reversed(ph) if c in CODE)


def synth(voice, ph, cfg=None):
    a = np.concatenate([c.audio_float_array for c in voice.synthesize(f"[[ {ph} ]]", cfg)])
    return a, voice.config.sample_rate


def trim(a, thr=0.01):
    idx = np.where(np.abs(a) > thr)[0]
    return a[idx[0] : idx[-1] + 1] if len(idx) else a


def save(path, a, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def textgrid(path, dur, sites):
    ivs, t = [], 0.0
    for lab, s, e in sites:
        if s > t:
            ivs.append(("", t, s))
        ivs.append((lab, s, e))
        t = e
    if t < dur:
        ivs.append(("", t, dur))
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {dur}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "sites"',
        "        xmin = 0",
        f"        xmax = {dur}",
        f"        intervals: size = {len(ivs)}",
    ]
    for i, (lab, s, e) in enumerate(ivs, 1):
        lines += [
            f"        intervals [{i}]:",
            f"            xmin = {s}",
            f"            xmax = {e}",
            f'            text = "{lab}"',
        ]
    path.write_text("\n".join(lines) + "\n")


def make(voice, chunks, which, cfg, wav, chunked_wav):
    """which[k] picks version 0/1 for chunk k; writes continuous wav + chunked wav/TextGrid."""
    parts, sites, t = [], [], 0.2
    for k, (lab, *ph) in enumerate(chunks):
        a, sr = synth(voice, ph[which], cfg)
        a = trim(a)
        if lab:
            sites.append((f"{lab}={value(ph[which])}", t, t + len(a) / sr))
        parts.append(a)
        t += len(a) / sr
    pad = np.zeros(int(0.2 * sr))
    audio = np.concatenate([pad] + parts + [pad])
    save(chunked_wav, audio, sr)
    textgrid(chunked_wav.with_suffix(".TextGrid"), len(audio) / sr, sites)
    a, sr = synth(voice, " ".join(c[1 + which] for c in chunks), cfg)
    save(wav, a, sr)


truth = {}
for sid, chunks in SENT.items():
    for vi, ver in enumerate("AB"):
        make(siwis, chunks, vi, None, corpus / f"model/{sid}_{ver}.wav", corpus / f"chunked/model/{sid}_{ver}.wav")

# P01, P03 imitate; P02 always produces B; P04 always produces A.
PARTICIPANTS = (
    ("P01", siwis, None, 0.9, {"A": 0, "B": 1}),
    ("P02", siwis, None, 1.2, {"A": 1, "B": 1}),
    ("P03", upmc, 0, 1.0, {"A": 0, "B": 1}),
    ("P04", upmc, 1, 1.1, {"A": 0, "B": 0}),
)
for part, voice, spk, scale, produce in PARTICIPANTS:
    cfg = SynthesisConfig(speaker_id=spk, length_scale=scale)
    for sid, chunks in SENT.items():
        for ver in "AB":
            name = f"{part}_{sid}_{ver}"
            make(
                voice,
                chunks,
                produce[ver],
                cfg,
                corpus / f"participants/{name}.wav",
                corpus / f"chunked/participants/{name}.wav",
            )
            truth[name] = {lab: value(ph[produce[ver]]) for lab, *ph in chunks if lab}
(corpus / "truth.json").write_text(json.dumps(truth, indent=1))
print("ok")
