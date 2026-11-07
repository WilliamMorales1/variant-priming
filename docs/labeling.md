# Labeling and mid-vowel charts

Tools for hand-labeling the model talker's sentence pairs and checking the vowel contrast. They are separate from `analyse_priming.py`: nothing here feeds the analysis.

```
labeler/
  denoise.sh          # writes <dir>_denoised/ next to a corpus dir, originals untouched
  serve.py            # local labeling server
  index.html          # labeling page
  labels_model.csv    # saved labels for corpus/model (git-ignored, created on first run)
plots/
  mid_vowel_chart.py  # F1/F2 charts of the mid vowels
```

## Denoising

```fish
labeler/denoise.sh corpus/model corpus/model_alt
```

Each `.wav` gets a 60 Hz high-pass and ffmpeg `afftdn` (12 dB reduction, adaptive noise tracking), at 16 kHz mono. `.lab` and `.TextGrid` files are copied, so the denoised directory is a drop-in copy with identical timing. The recordings were already quiet; the noise floor drops by 3 to 9 dB and speech level is unchanged.

## Labeling page

```fish
python labeler/serve.py corpus/model [port]    # default port 8000
```

Open `http://localhost:8000`. One page per sentence, with a card for each version (`v1`, `v2`): play button, transcript, and `+` / `-` buttons for `liaison` and `close_mid_vowel`. Each click autosaves to `labeler/labels_<dir>.csv`, and reopening resumes at the first incomplete pair. The checkbox switches between denoised and original audio when `<dir>_denoised` exists.

Keys: `1` / `2` play v1 / v2; `L` and `Shift+L` set liaison `+` / `-`; `V` and `Shift+V` set close-mid `+` / `-` (all act on the last version played); arrows change pair.

Columns: `file,sentence,version,text,liaison,close_mid_vowel`. Unlabeled cells stay empty.

## How the current labels were decided

- `liaison`: by ear.
- `close_mid_vowel`: by F1 of the word-final mid vowels (e, ɛ, o, ɔ, ø, œ) in each version. The aligner's phone labels give each vowel's lexical category. Each category pair (e/ɛ, o/ɔ, ø/œ) gets a reference axis from the pooled median F1 of its two categories, and each token is placed on it (0 = open reference, 1 = close reference). The version with the higher median position in a sentence gets `+`.

| sentence | `+` is | median gap (v2 minus v1) | tokens |
|---|---|---|---|
| S01 | v2 | +0.40 | 6 |
| S02 | v1 | -0.04 | 8 |
| S03 | v1 | -0.38 | 9 |
| S04 | v2 | +0.31 | 6 |
| S05 | v2 | +0.20 | 8 |

The two variables are labeled independently and do not have to agree. In S02 and S03 `close_mid_vowel` is `+` on v1 while `liaison` is `+` on v2. S02's gap is near zero, so its label is a toss-up. Limits: the TTS contrasts are small (the e/ɛ reference gap is about 70 Hz), the aligner intervals are rough, and F2 from this corpus is too noisy to use. Listen before trusting any single sentence.

## Mid-vowel charts

```fish
python plots/mid_vowel_chart.py [--final-only] [-o out.png]
```

One panel for v1 and one for v2. Needs matplotlib in the environment (`conda install -n variant-priming matplotlib`). Axes are reversed like a vowel quadrilateral (F2 on x, F1 on y). Each token is drawn as its IPA symbol, with a 1.5 sd ellipse per vowel (hidden under 4 tokens) and a filled marker at the mean. Tokens are vowels of 60 ms or more, F1/F2 taken as the median over the central half of the vowel and three formant ceilings (5000, 5500, 6000 Hz).

| option | effect |
|---|---|
| `--final-only` | drop vowels followed by another vowel in the same word (nonfinal syllables) |
| `--audio`, `-o` | override the audio directory and output path |

Output goes to `plots/` (git-ignored). Only the vertical axis (F1) is reliable; F2 spreads implausibly (e.g. /e/ from 500 to 2200 Hz) because the aligner intervals are rough.
