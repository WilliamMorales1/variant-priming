#!/usr/bin/env python3
"""Compare participant mid vowels with the two model versions of each sentence.

Corpus layout (see README.md):
    corpus/model/<sentence>_<version>.wav + .TextGrid   (two versions per sentence)
    corpus/participants/<participant>_<sentence>_<version>.wav (version = the one heard)
    corpus/speakers.csv                                    (optional formant ceilings)
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import torch
from parselmouth.praat import call
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from transformers import AutoFeatureExtractor, AutoModel

CHECKPOINT = "microsoft/wavlm-large"
LAYER = 24
SITE_TIER = "sites"
MODEL_SPEAKER = "model"
# Searched per speaker unless speakers.csv fixes one (Escudero et al. 2009).
CEILINGS = tuple(range(4500, 6501, 250))
MIN_REFERENCES = 2
# Loose F1 and F2 ranges (Hz) for reference vowels: male and female voices, coarticulated /u/.
REFERENCE_RANGES = {"i": ((0, 700), (1500, 3500)), "a": ((450, 1200), (900, 2300)), "u": ((0, 550), (400, 1600))}
BAND = 0.25
SILENCE_DB = 35.0
EDGE = 0.05
PAUSE = 0.25
MIN_SITE = 0.01
PERMUTATIONS = 10000

TALKER_RE = re.compile(r"^(?P<sentence>[^_]+)_(?P<version>[^_]+)$")
PARTICIPANT_RE = re.compile(r"^(?P<participant>[^_]+)_(?P<sentence>[^_]+)_(?P<version>[^_]+)$")


def is_vowel(site_id):
    return site_id[:1].upper() == "V"


def is_reference(site_id):
    return site_id[:1].upper() == "R"


def read_sites(path):
    tg = parselmouth.read(str(path))
    tier = next(
        (i for i in range(1, call(tg, "Get number of tiers") + 1) if call(tg, "Get tier name", i) == SITE_TIER), None
    )
    if tier is None or not call(tg, "Is interval tier", tier):
        sys.exit(f"{path}: needs an interval tier named '{SITE_TIER}'")
    sites = {}
    for i in range(1, call(tg, "Get number of intervals", tier) + 1):
        label = call(tg, "Get label of interval", tier, i).strip()
        if not label:
            continue
        site_id, _, value = label.partition("=")
        sites[site_id.strip()] = {
            "value": value.strip(),
            "start": call(tg, "Get start time of interval", tier, i),
            "end": call(tg, "Get end time of interval", tier, i),
        }
    return sites


def write_sites(path, duration, sites):
    tg = call("Create TextGrid", 0.0, duration, SITE_TIER, "")
    for site_id, s in sorted(sites.items(), key=lambda kv: kv[1]["start"]):
        for t in (s["start"], s["end"]):
            try:
                call(tg, "Insert boundary", 1, t)
            except parselmouth.PraatError:
                pass  # boundary already shared with a neighbouring site
        interval = call(tg, "Get interval at time", 1, (s["start"] + s["end"]) / 2)
        call(tg, "Set interval text", 1, interval, f"{site_id}={s['value']}")
    call(tg, "Save as text file", str(path))


def mfcc(sound):
    m = sound.to_mfcc(12, 0.015, 0.01)
    cep = m.to_array()[1:]
    return np.vstack([cep, np.gradient(cep, axis=1)]).T, m.xs()


class WavLM:
    """WavLM returns one hidden layer of the speech model as frame features."""

    def __init__(self):
        self.extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT)
        self.model = AutoModel.from_pretrained(CHECKPOINT).eval()
        self.rate = self.extractor.sampling_rate

    def __call__(self, sound):
        snd = sound if sound.sampling_frequency == self.rate else sound.resample(self.rate)
        inputs = self.extractor(snd.values[0].astype(np.float32), sampling_rate=self.rate, return_tensors="pt")
        with torch.inference_mode():
            h = self.model(inputs.input_values, output_hidden_states=True).hidden_states[LAYER][0]
        # 20 ms hop, ~25 ms receptive field.
        return h.numpy(), sound.xmin + 0.0125 + 0.02 * np.arange(len(h))


def speech_frames(times, level):
    """speech_frames returns the frame indices kept for alignment: the speech span minus long pauses."""
    speech = level > level.max() - SILENCE_DB
    idx = np.flatnonzero(speech)
    # A margin keeps quiet onsets and offsets (voiceless fricatives, released stops).
    keep = (times >= times[idx[0]] - EDGE) & (times <= times[idx[-1]] + EDGE)
    run_start = None
    for i in range(idx[0], idx[-1] + 2):
        if i <= idx[-1] and not speech[i]:
            run_start = i if run_start is None else run_start
        elif run_start is not None:
            inner = (times >= times[run_start] + EDGE) & (times <= times[i - 1] - EDGE)
            if times[i - 1] - times[run_start] >= PAUSE:
                keep &= ~inner
            run_start = None
    return np.flatnonzero(keep)


class Recording:
    """Recording caches the analyses of one wav file."""

    def __init__(self, wav, wavlm):
        self.sound = parselmouth.Sound(str(wav)).convert_to_mono()
        self.duration = self.sound.get_total_duration()
        self.wavlm = wavlm
        self._intensity = None
        self._formants = {}
        self._features = {}

    def vowel(self, site, ceiling):
        """vowel returns mean F1 and F2 (Hz) over the central half of the site."""
        if ceiling not in self._formants:
            self._formants[ceiling] = self.sound.to_formant_burg(0.005, 5, ceiling, 0.025, 50.0)
        # The same central half in model talker and participant keeps consonant transitions matched (same
        # word), so they cancel. Each token's steadiest half scored worse (44 vs 52/80): it measured
        # different parts of the vowel in different speakers. A frame median also scored worse.
        q = (site["end"] - site["start"]) / 4
        return tuple(
            call(self._formants[ceiling], "Get mean", n, site["start"] + q, site["end"] - q, "hertz") for n in (1, 2)
        )

    def features(self, extractor):
        """features returns per-utterance normalised frames, their times and the aligned frames."""
        if extractor not in self._features:
            if self._intensity is None:
                self._intensity = self.sound.to_intensity(75.0, 0.005)
            feats, times = extractor(self.sound)
            level = np.interp(times, self._intensity.xs(), self._intensity.values[0])
            keep = speech_frames(times, level)
            # Mean/variance normalisation removes much of the speaker difference.
            mu, sd = feats[keep].mean(axis=0), feats[keep].std(axis=0) + 1e-8
            self._features[extractor] = ((feats - mu) / sd, times, keep)
        return self._features[extractor]

    def speech(self):
        # Alignment uses MFCCs: WavLM's 20 ms hop placed boundaries worse on the test corpus.
        feats, times, keep = self.features(mfcc)
        return feats[keep], times[keep]

    def pooled(self, site):
        feats, times, _ = self.features(self.wavlm)
        idx = np.flatnonzero((times >= site["start"]) & (times <= site["end"]))
        if not len(idx):
            idx = [int(np.argmin(np.abs(times - (site["start"] + site["end"]) / 2)))]
        # Unit-normalised frames: cosine geometry suits WavLM better than raw Euclidean.
        f = feats[idx] / (np.linalg.norm(feats[idx], axis=1, keepdims=True) + 1e-8)
        return f.mean(axis=0)


def dtw_path(x, y):
    """dtw_path returns the best banded DTW path between frame sequences x and y and its mean cost."""
    n, m = len(x), len(y)
    cost = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1))
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    width = max(BAND * max(n, m), abs(n - m)) + 1
    for i in range(1, n + 1):
        centre = i * m / n
        for j in range(max(1, int(centre - width)), min(m, int(centre + width)) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    path.reverse()
    return np.array(path), float(acc[n, m]) / len(path)


def project_sites(talker, talker_sites, participant):
    """project_sites maps the model talker's site intervals onto the participant recording."""
    (tx, tt), (sx, st) = talker.speech(), participant.speech()
    path, cost = dtw_path(tx, sx)
    t_times, s_times = tt[path[:, 0]], st[path[:, 1]]
    # Average participant times per model-talker frame so the mapping is a function.
    uniq = np.unique(t_times)
    mapped = np.array([s_times[t_times == t].mean() for t in uniq])
    projected = {}
    for site_id, s in talker_sites.items():
        start = min(max(0.0, float(np.interp(s["start"], uniq, mapped))), participant.duration - MIN_SITE)
        end = min(max(float(np.interp(s["end"], uniq, mapped)), start + MIN_SITE), participant.duration)
        projected[site_id] = {"value": "?", "start": start, "end": end}
    return projected, cost


def project_both(versions, participant):
    """project_both averages the projections from both model versions."""
    # Projecting from the heard version alone pulled vowels toward it, even for non-imitators.
    (pa, ca), (pb, cb) = (project_sites(rec, sites, participant) for rec, sites in versions)
    projected = {}
    for k in pa.keys() & pb.keys():
        start = (pa[k]["start"] + pb[k]["start"]) / 2
        end = max((pa[k]["end"] + pb[k]["end"]) / 2, start + MIN_SITE)
        projected[k] = {"value": "?", "start": start, "end": end}
    return projected, (ca + cb) / 2


def load_ceilings(corpus):
    path = corpus / "speakers.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["speaker"]: float(row["formant_ceiling"]) for row in csv.DictReader(f)}


def load_talker(corpus, wavlm):
    talker = {}
    for wav in sorted((corpus / "model").glob("*.wav")):
        m = TALKER_RE.match(wav.stem)
        tg = wav.with_suffix(".TextGrid")
        if not m:
            sys.exit(f"{wav.name}: expected <sentence>_<version>.wav")
        if not tg.exists():
            sys.exit(f"{wav.name}: missing {tg.name}")
        talker.setdefault(m["sentence"], {})[m["version"]] = (Recording(wav, wavlm), read_sites(tg))
    for sentence, versions in talker.items():
        if len(versions) != 2:
            sys.exit(f"sentence {sentence}: need exactly 2 model versions, found {sorted(versions)}")
        (a, (_, sa)), (b, (_, sb)) = versions.items()
        if sa.keys() != sb.keys():
            print(
                f"warning: {sentence}: sites differ between {a} and {b}; using {sorted(sa.keys() & sb.keys())}",
                file=sys.stderr,
            )
        n_ref = sum(is_reference(k) for k in sa.keys() & sb.keys())
        if n_ref < MIN_REFERENCES:
            sys.exit(
                f"sentence {sentence}: has {n_ref} R sites, needs {MIN_REFERENCES} "
                f"(2-3 clear a/i/u vowels per sentence)"
            )
    return talker


def plausible_reference(label, f1, f2, where=None):
    """plausible_reference reports whether a reference vowel's formants fit its label."""
    if math.isnan(f1) or math.isnan(f2):
        return False
    ranges = REFERENCE_RANGES.get(label)
    ok = ranges is None or all(lo <= f <= hi for f, (lo, hi) in zip((f1, f2), ranges))
    if not ok and where:
        # Usually a misplaced interval or a formant-tracking error; one such token skews Lobanov.
        print(f"drop reference {where}={label}: F1={f1:.0f} F2={f2:.0f} Hz", file=sys.stderr)
    return ok


def choose_ceiling(refs, targets=None):
    """choose_ceiling picks the ceiling that makes each reference vowel's tokens most consistent.

    With targets (the model talker's log formants per site), each token is compared with the model talker's
    token of the same word, so the neighbouring consonants' effect cancels.
    """
    best, best_score = CEILINGS[len(CEILINGS) // 2], math.inf
    for ceiling in CEILINGS:
        by_label, bad = {}, 0
        for rec, site, label, key in refs:
            f = rec.vowel(site, ceiling)
            if plausible_reference(label, *f) and (targets is None or key in targets):
                by_label.setdefault(label, []).append(np.log(f) - (targets[key] if targets is not None else 0))
            else:
                bad += 1
        # Log formants make the spread scale-free; implausible tokens count as large errors.
        spread = sum(((np.array(v) - np.mean(v, axis=0)) ** 2).sum() for v in by_label.values())
        score = (spread + bad) / max(len(refs), 1)
        if score < best_score:
            best, best_score = ceiling, score
    return best


def reference_stats(tokens, keys=None):
    """reference_stats returns a speaker's F1/F2 centre and spread from reference vowels only.

    tokens holds (label, site key, formants); keys restricts them to a given set of sites.
    """
    by_site = {}
    for label, key, f in tokens:
        if keys is None or key in keys:
            by_site.setdefault((label, key), []).append(f)
    labels = {label for label, _ in by_site}
    if sum(len(v) for v in by_site.values()) < 3 or len(labels) < 2:
        return None
    # Each label weighs the same, and each site the same within its label, so an extra [a] or a
    # site said twice does not pull the centre.
    n_sites = {lab: sum(1 for l, _ in by_site if l == lab) for lab in labels}
    w = np.concatenate([np.full(len(v), 1 / (len(v) * n_sites[l])) for (l, _), v in by_site.items()])
    arr = np.concatenate([np.array(v) for v in by_site.values()])
    mu = (w[:, None] * arr).sum(axis=0) / w.sum()
    sd = np.sqrt((w[:, None] * (arr - mu) ** 2).sum(axis=0) / w.sum())
    return (mu, sd) if (sd > 0).all() else None


def contrast_directions(talker, feature):
    """contrast_directions pools the model talker's A-minus-B differences per vowel pair.

    Each pair gets a unit direction, the mean contrast size along it, and the spread of one
    model-talker token around its pooled target.
    """
    diffs = {}
    for sentence, versions in talker.items():
        (va_name, (_, sa)), (vb_name, (_, sb)) = versions.items()
        for k in sa.keys() & sb.keys():
            va, vb = sa[k]["value"], sb[k]["value"]
            if not is_vowel(k) or va == vb:
                continue
            pair = tuple(sorted((va, vb)))
            fa, fb = feature(sentence, va_name, k), feature(sentence, vb_name, k)
            if fa is None or fb is None:
                continue
            d = fa - fb
            diffs.setdefault(pair, []).append(d if va == pair[0] else -d)
    dirs = {}
    for pair, ds in diffs.items():
        ds = np.array(ds)
        u = ds.mean(axis=0) / (np.linalg.norm(ds.mean(axis=0)) + 1e-8)
        sizes = ds @ u
        noise = float(np.std(sizes, ddof=1) / 2) if len(ds) > 1 else math.nan
        if len(ds) < 2:
            print(
                f"warning: vowel pair {pair[0]}/{pair[1]} has 1 sentence; its direction is that "
                "sentence's own difference",
                file=sys.stderr,
            )
        dirs[pair] = (u, float(sizes.mean()), noise)
    return dirs


def place(fs, fh, fo, value_heard, value_other, dirs):
    """place locates the participant token on the model talker's contrast: 0 = other version, 1 = heard."""
    pair = tuple(sorted((value_heard, value_other)))
    if pair not in dirs or fs is None or fh is None or fo is None:
        return math.nan, math.nan, False
    u, size, noise = dirs[pair]
    sign = 1.0 if value_heard == pair[0] else -1.0
    rel = fs - (fh + fo) / 2
    x = sign * float(rel @ u)
    off = float(np.linalg.norm(rel - (rel @ u) * u)) / size
    # Within one model-talker token spread of the midpoint, the two versions are not distinguishable.
    uncertain = not math.isnan(noise) and abs(x) < noise
    return 0.5 + x / size, off, uncertain


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum(np.mean([x == lab for x in a]) * np.mean([y == lab for y in b]) for lab in labels)
    return (po - pe) / (1 - pe) if pe < 1 else math.nan


def analyse(corpus, out, metric):
    fixed = load_ceilings(corpus)
    wavlm = WavLM() if metric == "vector" else None
    talker = load_talker(corpus, wavlm)
    out.mkdir(parents=True, exist_ok=True)

    participants = []
    for wav in sorted((corpus / "participants").glob("*.wav")):
        m = PARTICIPANT_RE.match(wav.stem)
        if not m:
            print(f"skip {wav.name}: expected <participant>_<sentence>_<version>.wav", file=sys.stderr)
            continue
        participant, sentence, heard = m["participant"], m["sentence"], m["version"]
        if sentence not in talker or heard not in talker[sentence]:
            print(f"skip {wav.name}: no model version {sentence}_{heard}", file=sys.stderr)
            continue
        other = next(v for v in talker[sentence] if v != heard)
        rec = Recording(wav, wavlm)
        manual = wav.with_suffix(".TextGrid")
        if manual.exists():
            sites, source, cost = read_sites(manual), "manual", math.nan
        else:
            sites, cost = project_both([talker[sentence][heard], talker[sentence][other]], rec)
            source = "auto"
            write_sites(wav.with_suffix(".auto.TextGrid"), rec.duration, sites)
        participants.append(
            {
                "participant": participant,
                "sentence": sentence,
                "heard": heard,
                "other": other,
                "rec": rec,
                "sites": sites,
                "textgrid": source,
                "align_cost": cost,
                "file": wav.stem,
            }
        )

    # Reference tokens per speaker: (recording, site, label, (sentence, site id), where).
    refs = {}
    for sentence, versions in talker.items():
        for version, (rec, sites) in versions.items():
            for k, s in sites.items():
                if is_reference(k):
                    refs.setdefault(MODEL_SPEAKER, []).append(
                        (rec, s, s["value"], (sentence, k), f"{sentence}_{version}:{k}")
                    )
    for st in participants:
        sh = talker[st["sentence"]][st["heard"]][1]
        for k, s in st["sites"].items():
            if is_reference(k) and k in sh:
                refs.setdefault(st["participant"], []).append(
                    (st["rec"], s, sh[k]["value"], (st["sentence"], k), f"{st['file']}:{k}")
                )
    ceilings, kept = {}, {}

    def keep_references(speaker):
        kept[speaker] = []
        for rec, s, label, key, where in refs[speaker]:
            f1, f2 = rec.vowel(s, ceilings[speaker])
            if plausible_reference(label, f1, f2, where):
                kept[speaker].append((label, key, np.array([f1, f2])))

    if MODEL_SPEAKER in refs:
        ceilings[MODEL_SPEAKER] = fixed.get(MODEL_SPEAKER) or choose_ceiling([t[:4] for t in refs[MODEL_SPEAKER]])
        keep_references(MODEL_SPEAKER)
    targets = {}
    for _, key, f in kept.get(MODEL_SPEAKER, []):
        targets.setdefault(key, []).append(np.log(f))
    targets = {key: np.mean(v, axis=0) for key, v in targets.items()}
    for speaker in refs.keys() - {MODEL_SPEAKER}:
        ceilings[speaker] = fixed.get(speaker) or choose_ceiling([t[:4] for t in refs[speaker]], targets)
        keep_references(speaker)
    # Both sides of a comparison are normalised over the same reference words, so consonant
    # effects on those vowels (e.g. a lowered F2 after uvular /R/) weigh the same in both.
    stats, talker_stats = {}, {}
    for speaker, toks in kept.items():
        if speaker == MODEL_SPEAKER:
            continue
        shared = {key for _, key, _ in toks} & set(targets)
        stats[speaker] = reference_stats(toks, shared)
        talker_stats[speaker] = reference_stats(kept[MODEL_SPEAKER], shared)

    def z_score(f, st):
        return None if st is None or np.isnan(f).any() else (f - st[0]) / st[1]

    talker_f = {}
    for sentence, versions in talker.items():
        for version, (rec, sites) in versions.items():
            for k, s in sites.items():
                if is_vowel(k):
                    talker_f[(sentence, version, k)] = np.array(
                        rec.vowel(s, ceilings.get(MODEL_SPEAKER, CEILINGS[len(CEILINGS) // 2]))
                    )
    talker_z, formant_dirs = {}, {}
    for speaker, tst in talker_stats.items():
        talker_z[speaker] = {key: z_score(f, tst) for key, f in talker_f.items()}
        formant_dirs[speaker] = contrast_directions(talker, lambda s, v, k, tz=talker_z[speaker]: tz.get((s, v, k)))
    vector_dirs = (
        contrast_directions(talker, lambda s, v, k: talker[s][v][0].pooled(talker[s][v][1][k])) if wavlm else {}
    )

    costs = np.array([st["align_cost"] for st in participants if not math.isnan(st["align_cost"])])
    med = np.median(costs) if len(costs) else math.nan
    mad = np.median(np.abs(costs - med)) if len(costs) else math.nan

    rows = []
    for st in participants:
        sentence, heard, other = st["sentence"], st["heard"], st["other"]
        t_heard, sites_heard = talker[sentence][heard]
        t_other, sites_other = talker[sentence][other]
        cost = st["align_cost"]
        for site_id in sorted(sites_heard.keys() & sites_other.keys() & st["sites"].keys()):
            sh, so, ss = sites_heard[site_id], sites_other[site_id], st["sites"][site_id]
            if not is_vowel(site_id) or sh["value"] == so["value"]:
                continue  # liaison sites are projected into the TextGrid for listening only
            r = {
                "participant": st["participant"],
                "sentence": sentence,
                "heard": heard,
                "other": other,
                "site": site_id,
                "value_heard": sh["value"],
                "value_other": so["value"],
                "textgrid": st["textgrid"],
                "start": ss["start"],
                "end": ss["end"],
                "duration": ss["end"] - ss["start"],
                "value_coded": "" if ss["value"] in ("", "?") else ss["value"],
                "ceiling": ceilings.get(st["participant"], ""),
                "align_cost": cost,
                # Far above the corpus's usual cost: restart, hesitation or a misplaced projection.
                "check_alignment": bool(cost > med + 4 * 1.4826 * mad) if not math.isnan(cost) else "",
            }
            p = st["participant"]
            f = np.array(st["rec"].vowel(ss, ceilings.get(p, CEILINGS[len(CEILINGS) // 2])))
            z = z_score(f, stats.get(p))
            fh, fo = talker_f[(sentence, heard, site_id)], talker_f[(sentence, other, site_id)]
            tz = talker_z.get(p, {})
            zh, zo = tz.get((sentence, heard, site_id)), tz.get((sentence, other, site_id))
            r["F1"], r["F2"] = f
            r["F1_heard"], r["F2_heard"], r["F1_other"], r["F2_other"] = *fh, *fo
            if z is not None:
                r["z1"], r["z2"] = z
            vu = False
            r["formant_position"], r["formant_off_axis"], fu = place(
                z, zh, zo, sh["value"], so["value"], formant_dirs.get(p, {})
            )
            if wavlm:
                r["vector_position"], _, vu = place(
                    st["rec"].pooled(ss), t_heard.pooled(sh), t_other.pooled(so), sh["value"], so["value"], vector_dirs
                )
            pos = r[f"{metric}_position"]
            if math.isnan(pos):
                rows.append(r)
                continue  # too few reference vowels for this speaker to normalise
            r["position"] = pos
            r["uncertain"] = fu if metric == "formant" else vu
            r["value_estimated"] = r["value_heard"] if pos > 0.5 else r["value_other"]
            r["matches_heard"] = pos > 0.5
            if r["value_coded"]:
                r["coded_agrees_auto"] = r["value_coded"] == r["value_estimated"]
            rows.append(r)

    write_csv(
        out / "sites.csv",
        [
            "participant",
            "sentence",
            "heard",
            "other",
            "site",
            "value_heard",
            "value_other",
            "textgrid",
            "start",
            "end",
            "duration",
            "align_cost",
            "check_alignment",
            "ceiling",
            "F1",
            "F2",
            "F1_heard",
            "F2_heard",
            "F1_other",
            "F2_other",
            "z1",
            "z2",
            "formant_position",
            "formant_off_axis",
            "vector_position",
            "position",
            "uncertain",
            "value_estimated",
            "matches_heard",
            "value_coded",
            "coded_agrees_auto",
        ],
        rows,
    )
    rows = [r for r in rows if r["value_coded"] or "value_estimated" in r]
    return fit_model(rows, out)


def oriented(r):
    """oriented codes a token against its site's alphabetically first version."""
    ref, ref_value = min((r["heard"], r["value_heard"]), (r["other"], r["value_other"]))
    produced = r["value_coded"] or r["value_estimated"]
    pos = r.get("position", math.nan)
    return {
        "participant": r["participant"],
        "item": f"{r['sentence']}:{r['site']}",
        "y": float(produced == ref_value),
        "heard": 0.5 if r["heard"] == ref else -0.5,
        "pos": pos if r["heard"] == ref else 1 - pos,
        "match": produced == r["value_heard"],
        "coded": bool(r["value_coded"]),
        "uncertain": r.get("uncertain") is True,
    }


def within_effect(df, col, rng):
    """within_effect averages each participant's condition difference and permutes heard within participants."""
    groups = [
        (g[col].to_numpy(), g["heard"].to_numpy())
        for _, g in df.groupby("participant")
        if g["heard"].nunique() == 2 and g[col].notna().all()
    ]
    if not groups:
        return math.nan, math.nan, {}

    def stat(x, h):
        return x[h > 0].mean() - x[h < 0].mean()

    obs = np.mean([stat(x, h) for x, h in groups])
    perm = np.mean([[stat(x, rng.permutation(h)) for _ in range(PERMUTATIONS)] for x, h in groups], axis=0)
    p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (PERMUTATIONS + 1)
    per = {
        name: stat(g[col].to_numpy(), g["heard"].to_numpy())
        for name, g in df.groupby("participant")
        if g["heard"].nunique() == 2
    }
    return obs, p, per


def fit_model(rows, out):
    """fit_model writes the priming tests: a mixed logistic regression and within-participant permutations."""
    df = pd.DataFrame(
        [oriented(r) for r in rows], columns=["participant", "item", "y", "heard", "pos", "match", "coded", "uncertain"]
    )
    if df["participant"].nunique() < 2 or df["heard"].nunique() < 2:
        return "no model: needs 2+ participants and both versions heard"
    vc = {"participant": "0 + C(participant)", "item": "0 + C(item)", "slope": "0 + C(participant):heard"}
    fit = BinomialBayesMixedGLM.from_formula("y ~ heard", vc, df).fit_vb()
    b, sd = fit.fe_mean[1], fit.fe_sd[1]
    z = b / sd
    slopes = {
        name.split("[")[1].split("]")[0]: b + mean
        for name, mean in zip(fit.model.vc_names, fit.vc_mean)
        if ":heard" in name
    }
    rng = np.random.default_rng(0)
    dy, py, per_y = within_effect(df, "y", rng)
    auto = df[~df["coded"]]
    dp, pp, per_p = within_effect(auto, "pos", rng) if len(auto) else (math.nan, math.nan, {})
    lines = [
        "Mixed-effects logistic regression (variational Bayes, statsmodels)",
        "produced_ref ~ heard_ref + (1 + heard_ref | participant) + (1 | item)",
        (
            f"n = {len(df)} tokens ({df['coded'].sum()} coded by ear, {df['uncertain'].sum()} "
            f"uncertain), {df['participant'].nunique()} participants, {df['item'].nunique()} items"
        ),
        "",
        (
            f"priming slope   b = {b:+.3f}  sd = {sd:.3f}  z = {z:+.2f}  "
            f"p(two-sided) = {math.erfc(abs(z) / math.sqrt(2)):.4f}  odds ratio = {math.exp(b):.2f}"
        ),
        f"intercept       b = {fit.fe_mean[0]:+.3f}  sd = {fit.fe_sd[0]:.3f}",
        "",
        *(
            f"random sd {name:<12} {math.exp(mean):.3f}  (log-sd posterior sd {s:.3f})"
            for name, mean, s in zip(fit.model.vcp_names, fit.vcp_mean, fit.vcp_sd)
        ),
        "",
        f"Within-participant permutation tests ({PERMUTATIONS} shuffles of heard within participant)",
        f"produced_ref rate, heard ref - heard other: {dy:+.3f}  p = {py:.4f}",
        f"position toward ref (acoustic tokens only):  {dp:+.3f}  p = {pp:.4f}",
        "",
        "participant  slope (log-odds)  rate diff  position diff  matches heard",
        *(
            f"{p:>11}  {slopes[p]:+16.3f}  {per_y.get(p, math.nan):+9.3f}  "
            f"{per_p.get(p, math.nan):+13.3f}  {g['match'].sum()}/{len(g)}"
            for p, g in df.groupby("participant")
        ),
    ]
    coded = [r for r in rows if r["value_coded"] and "value_estimated" in r]
    if coded:
        a = [r["value_coded"] for r in coded]
        e = [r["value_estimated"] for r in coded]
        lines += [
            "",
            (
                f"ear vs machine: {sum(x == y for x, y in zip(a, e))}/{len(coded)} agree, "
                f"Cohen's kappa = {kappa(a, e):.3f}"
            ),
        ]
    text = "\n".join(lines) + "\n"
    (out / "model.txt").write_text(text, encoding="utf-8")
    return text


def write_csv(path, cols, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: round(v, 4) if isinstance(v, float) else v for c, v in ((c, r.get(c, "")) for c in cols)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path, nargs="?", default=Path("corpus"))
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    ap.add_argument(
        "--metric",
        choices=("vector", "formant"),
        default="vector",
        help="WavLM pooled contrast vector, or Lobanov F1/F2 (fast, no model download)",
    )
    args = ap.parse_args()
    print(analyse(args.corpus, args.out, args.metric))
    print(f"wrote {args.out}/sites.csv, model.txt")


if __name__ == "__main__":
    main()
