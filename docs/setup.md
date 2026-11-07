# Setup

Nothing is installed as a package; the script runs from the repo root. Needs Python 3.10+.

**conda** ([Miniforge](https://github.com/conda-forge/miniforge)), what CI uses. conda-forge picks the CPU build of PyTorch on machines without an NVIDIA GPU:

```fish
conda env create -f environment.yml
conda activate variant-priming
```

**venv + pip** (pip 25.1+ for `--group`). Install the CPU PyTorch wheel (~200 MB) first, or pip pulls the CUDA one (~3 GB):

```fish
python -m venv .venv
source .venv/bin/activate.fish     # bash/zsh: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install . --group dev          # drop --group dev to skip pytest and ruff
```

The first `--metric vector` run downloads `microsoft/wavlm-large` (about 1.3 GB) into the Hugging Face cache. After that, `HF_HUB_OFFLINE=1` skips the network check. `--metric formant` loads no model.

Charts in `plots/` also need matplotlib (`conda install matplotlib`); it is not pinned in `pyproject.toml` because the analysis does not use it. Denoising needs `ffmpeg`.

`review_sites.praat` needs the [Praat](https://www.fon.hum.uva.nl/praat/) app (6.1 or later).

## Changing dependencies

Update both `pyproject.toml` and `environment.yml`.

## Tests

```fish
python -m pytest -q
ruff check . && ruff format --check .
```

`tests/test_analyse_priming.py` covers DTW, pause removal, TextGrid round trips, reference-vowel checks, Lobanov weights, ceiling search, contrast directions, kappa, the model, and two end-to-end runs (`--metric formant`) on a small corpus of Klatt-synthesized vowels: one participant imitates the version heard, one always says [e], and a hand-corrected TextGrid must override the projection. CI runs both commands on every push and pull request.

The scripts `tests/gen_fixture.py`, `map_fixture.py`, `reshape.py` and `score.py` build and score the larger TTS corpus; see [Validation](validation.md).
