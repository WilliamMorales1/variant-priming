# Validation

## Synthetic corpus

French TTS (piper) with 6 sentences × 2 versions, 10 mid-vowel sites and 2-3 `R` sites per sentence. There are 4 fake participants: two are the model talker voice with Praat-shifted formants, two are a second TTS voice. Two imitate the heard version, one always produces B and one always produces A. That gives 80 participant vowel tokens. Answering "heard" every time scores 60/80, so `tests/score.py` also prints accuracy on the 20 tokens where the truth is not the heard value, and the balanced mean.

| metric | projection | correct /80 | balanced | group p | fixed producers' slopes |
|---|---|---|---|---|---|
| WavLM | heard only | 59 | 0.71 | 0.004 | +1.06, +1.03 |
| WavLM (final) | both | 55 | 0.71 | 0.27 | +0.05, +0.05 |
| formant (old) | heard only | 56 | 0.63 | 0.0001 | +2.23, +1.76 |
| formant (final) | both | 46 | 0.60 | 1.00 | -0.09, 0.00 |

With the true productions, the model gives slopes of +12.8 for both imitators and +0.4 for both fixed producers (group p = 0.24). The formant metric is near chance on this TTS corpus with every setting tried (the TTS model talker's contrasts are tiny); the formant changes are kept for what they fix, not for a measured gain. Judge methods by `value_estimated` accuracy, not by boundary error, and check that the fixed producers' slopes stay near 0.

The corpus is not deterministic: compare methods on one build.

## Rebuilding it

Needs a second conda env with [piper-tts](https://github.com/OHF-Voice/piper1-gpl) 1.8 (`tts` below). `$D` is any scratch directory.

1. Download the voices into `$D/voices`:
   ```fish
   cd $D/voices; python -m piper.download_voices fr_FR-siwis-medium fr_FR-upmc-medium
   ```
2. From the repo root, in this order (`gen_fixture.py` must come before `reshape.py`, since the mapping needs unshifted audio):
   ```fish
   conda run -n tts python tests/gen_fixture.py $D $D/corpus2
   conda run -n variant-priming python tests/map_fixture.py $D/corpus2
   conda run -n variant-priming python tests/reshape.py $D/corpus2
   conda run -n variant-priming python analyse_priming.py $D/corpus2 -o $D/r
   python tests/score.py $D/corpus2 $D/r
   ```
