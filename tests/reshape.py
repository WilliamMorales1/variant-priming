"""Turn the participants in the model talker's voice into different 'speakers' with Praat Change gender."""

import sys
from pathlib import Path

import parselmouth
from parselmouth.praat import call

c = Path(sys.argv[1]) / "participants"
for part, formant_ratio, pitch in (("P01", 1.12, 260.0), ("P02", 0.88, 120.0)):
    for wav in c.glob(f"{part}_*.wav"):
        snd = parselmouth.Sound(str(wav))
        out = call(snd, "Change gender", 75, 600, formant_ratio, pitch, 1.0, 1.0)
        out.save(str(wav), "WAV")
