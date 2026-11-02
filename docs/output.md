# Output

## Placing a vowel

Each participant vowel is placed along the model talker's contrast direction for its vowel pair. The direction is the model talker's mean A − B difference, averaged over all sites with the same vowel pair. The position is measured from the site's model talker midpoint in units of the pooled contrast size: 0 = other version, 1 = heard.

- `--metric vector` (default): layer 24 of WavLM-large, frames unit-normalized and mean-pooled over the site (pooled phonological vector, after Choi et al. 2026, *[b] = [d] − [t] + [p]*).
- `--metric formant`: F1/F2 (Praat Burg, central 50% of the vowel), Lobanov-normalized from `R` tokens.

F1/F2 are measured with either metric; the metric only decides `value_estimated`.

Neighbouring consonants shift formants (uvular `r` lowers F2, `z`/`j` raise it). Since model talker and participant say the same word, those shifts cancel if both are measured and normalized the same way: the same central half of the vowel, a ceiling chosen by comparing each participant `R` vowel with the model talker's in the same word, and normalization of the participant and the model talker over the same `R` words (the ones the participant kept). Dropping the participants' `R` vowels next to `r` flipped 4/80 decisions this way, against 15/80 with per-speaker pools. This does not cover a participant who uses different consonants (an English `r`): listen to sites next to `r`.

## `results/sites.csv`

One row per participant × vowel site.

| column | meaning |
|---|---|
| `start`, `end`, `duration` | site interval in the participant file (s). Below about 50 ms, the central half spans less than two 25 ms analysis windows, so F1/F2 are shaky |
| `textgrid` | `auto` (projected) or `manual` (hand-corrected) |
| `align_cost`, `check_alignment` | mean DTW cost; flagged when above the corpus median + 4 scaled MAD (restart, hesitation, bad projection) |
| `ceiling` | formant ceiling used for that speaker |
| `F1`, `F2`, `F1_heard`, … `F2_other` | Hz, participant and both model versions |
| `z1`, `z2` | participant Lobanov z-scores |
| `position`, `formant_position`, `vector_position` | 0 = typical other-version vowel, 1 = typical heard-version vowel (`vector_*` empty with `--metric formant`) |
| `uncertain` | closer to the midpoint than one model-talker token's own spread; code these by ear first |
| `formant_off_axis` | distance from the contrast line (same units); large means a third vowel quality |
| `value_estimated`, `matches_heard` | heard value if `position` > 0.5, else the other value |
| `value_coded`, `coded_agrees_auto` | code by ear from a hand-corrected TextGrid |

## `results/model.txt`

Mixed-effects logistic regression: produced variant ~ version heard, with a random intercept and slope per participant and a random intercept per item (sentence + site). Each site is coded against its alphabetically first version, so the slope compares conditions within a participant. It uses `value_coded` where present, otherwise `value_estimated`. A positive `priming slope` means participants produce a variant more often after hearing it. Fitted by variational Bayes (statsmodels `BinomialBayesMixedGLM`), so p is approximate.

Also listed:

- each participant's slope (near 0 for a participant who always says the same thing, whatever their match rate) and match count;
- two within-participant permutation tests (10,000 shuffles of the heard version inside each participant), on the produced variant and on `position`; they do not rely on the random-slope variance;
- with codes by ear, agreement with `value_estimated` and Cohen's kappa.

## Limits

- The automatic estimate is a screening tool. It leans toward the heard version (non-imitators got slopes of +0.5 to +1.7 on test data). Code vowels by ear (`value_coded`) for the final model and report ear/machine agreement.
- Projecting from both model versions removed the lean toward the heard version on the synthetic corpus, but run the final model on codes by ear anyway.
- Listen to files flagged `check_alignment`.
