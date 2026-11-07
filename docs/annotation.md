# Annotation

## Corpus layout

```
corpus/
  model/S01_v1.wav  S01_v1.TextGrid               # two versions per sentence
  model/S01_v2.wav  S01_v2.TextGrid
  participants/P01_S01_v1.wav                    # participant_sentence_versionHeard
  speakers.csv                                    # optional: speaker,formant_ceiling
```

No underscores inside IDs (`S01`, `P01`, `v1`, `v2`). Recordings and TextGrids are git-ignored: they are participant data.

## Model talker TextGrids

Once per recording, in Praat: add an interval tier named `sites` and label each variable site `ID=value`. Use the same IDs in both versions of a sentence.

| prefix | type | interval covers | example labels |
|---|---|---|---|
| `V` | mid vowel | the vowel | `V1=e` / `V1=E` |
| `R` | reference vowel | a clear `a`, `i` or `u`, same word in both versions | `R1=a` |
| other, e.g. `L` | liaison, not analysed | the word junction | `L1=z` / `L1=0` |

Only `V` sites whose value differs between the two versions are tested. `R` sites are not tested: they alone set each speaker's formant normalization (Lobanov, each label weighted equally) and ceiling, so a participant's own variant choices cannot shift them. Every sentence needs 2 or 3, ideally covering `a`, `i` and `u` across the corpus; the script stops if a sentence has fewer than 2. An `R` token whose F1/F2 is impossible for its label (e.g. `i` with F1 > 700 Hz, usually a misplaced interval) is dropped from normalization with a message. Other sites are still projected into the participant's `.auto.TextGrid`, which makes the liaisons quick to find when listening.

## Formant ceilings

Without `speakers.csv`, the Praat formant ceiling is searched from 4500 to 6500 Hz per speaker: the one that makes each reference vowel's tokens most consistent wins. For a participant, each `R` token is compared with the model talker's token of the same word, so consonant effects cancel. To fix a ceiling, add a row; the model talker's row uses speaker `model`:

```
speaker,formant_ceiling
model,5500
P03,5000
```

## Participant files

The script aligns each participant recording with both model versions (DTW on mean/variance-normalized MFCCs, pauses of 250 ms or more removed), projects the model talker's sites from each, averages them, and writes `participants/<name>.auto.TextGrid`. Projecting from the heard version alone made non-imitators look primed.

A hand-corrected `participants/<name>.TextGrid` takes precedence. Replacing `?` on a `V` site with a code by ear (`V1=E`) adds that code to the output as `value_coded`, which the model uses instead of the automatic estimate.

DTW assumes the participant repeats the whole sentence. Pauses are handled, restarts are not: cut false starts out of the file.

## Reviewing participant files

`review_sites.praat` (open it in Praat, then Run) walks through the participant files that have no hand-corrected TextGrid yet. Each opens with formant tracks at that speaker's ceiling from `results/sites.csv`, louder for listening, with its flags (`align` for a high alignment cost, `V1?` for an uncertain vowel). Save writes `<name>.TextGrid`; Skip moves on; Stop ends, and the next run resumes where you left off.

| option | effect |
|---|---|
| `Flagged_only` | only files with `check_alignment` or an `uncertain` vowel |
| `Redo_manual` | also reopen files that already have a `<name>.TextGrid` |
| `Open_editor` off | only list the queue in the Info window |

Adapted from Will Styler's [`textgrid_revisor.praat`](https://github.com/stylerw/styler_praat_scripts).
