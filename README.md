# variant-priming

![Tests](https://github.com/WilliamMorales1/variant-priming/actions/workflows/tests.yml/badge.svg)

Does hearing one variant of a sentence prime a listener to produce that variant? Participants repeat sentences after hearing one of two recordings by a model talker. The two versions differ in phonetic variables, such as French nonfinal mid vowels (/e/\~/ɛ/, /o/\~/ɔ/, /ø/\~/œ/) or optional liaison. `analyse_priming.py` aligns every repetition with both versions, places each vowel between the two model variants, and fits a mixed-effects model of produced variant ~ version heard. Liaison and other non-vowel sites are coded by ear; the script marks where each one falls.

Any language works as long as the variable sites are vowel-quality contrasts with a clear `a`, `i` or `u` nearby for normalization. [`sentences.txt`](sentences.txt) has example French stimuli, and [`method.pdf`](method.pdf) the method with its equations and weaknesses.

## Quick start

```fish
conda env create -f environment.yml
conda activate variant-priming
python analyse_priming.py corpus -o results                    # WavLM metric, downloads microsoft/wavlm-large (1.3 GB)
python analyse_priming.py corpus -o results --metric formant   # F1/F2 only: fast, no model
```

1. Record the model talker's two versions of each sentence and mark the sites in Praat ([annotation](docs/annotation.md)).
2. Record each participant repeating the version they heard, and run the script. It writes each participant's projected sites to `participants/<name>.auto.TextGrid`.
3. Check the projections with `review_sites.praat`, coding vowels by ear where unsure ([review](docs/annotation.md#reviewing-participant-files)).
4. Rerun. Read `results/model.txt` for the test and `results/sites.csv` for every token ([output](docs/output.md)).

A run peaks at about 2 GB of RAM and runs on a laptop CPU.

## Docs

- [Setup](docs/setup.md): conda, pip, CPU-only PyTorch, tests.
- [Annotation](docs/annotation.md): corpus layout, model-talker TextGrids, formant ceilings, reviewing participant files.
- [Output](docs/output.md): how a vowel is placed, `sites.csv` columns, `model.txt`, limits.
- [Validation](docs/validation.md): results on the synthetic TTS corpus and how to rebuild it.

## License

MIT
