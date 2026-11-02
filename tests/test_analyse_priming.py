import math

import numpy as np
import parselmouth
import pytest
from parselmouth.praat import call

import analyse_priming as ap

RATE = 16000
# Male-ish targets (F1, F2) in Hz; participants get a longer vocal tract via a scale factor.
VOWELS = {"a": (750, 1300), "i": (300, 2300), "u": (320, 800), "e": (400, 2000), "E": (580, 1750)}


def vowel(label, scale=1.0, f0=120, dur=0.16):
    f1, f2 = (f * scale for f in VOWELS[label])
    k = call("Create KlattGrid from vowel", label, dur, f0, f1, 80, f2, 90, 2600 * scale, 120, 3500 * scale, 0.05, 1000)
    return call(call(k, "To Sound"), "Resample", RATE, 50).values[0]


def utterance(labels, scale=1.0, f0=120):
    """utterance concatenates vowels between quiet gaps and returns the sound and each vowel's interval."""
    rng = np.random.default_rng(0)
    gap = 1e-4 * rng.standard_normal(int(0.05 * RATE))
    parts, spans, t = [1e-4 * rng.standard_normal(int(0.2 * RATE))], [], 0.2
    for lab in labels:
        v = vowel(lab, scale, f0)
        spans.append((t, t + len(v) / RATE))
        parts += [v, gap]
        t += (len(v) + len(gap)) / RATE
    parts.append(1e-4 * rng.standard_normal(int(0.2 * RATE)))
    return parselmouth.Sound(np.concatenate(parts), RATE), spans


# Sentence: (site ids, version A vowels, version B vowels).
SENTENCES = {
    "S01": (["R1", "V1", "R2"], ["a", "e", "i"], ["a", "E", "i"]),
    "S02": (["R1", "V1", "R2", "R3"], ["i", "E", "a", "u"], ["i", "e", "a", "u"]),
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    (root / "model").mkdir()
    (root / "participants").mkdir()
    for sentence, (ids, va, vb) in SENTENCES.items():
        for version, labels in (("A", va), ("B", vb)):
            snd, spans = utterance(labels)
            snd.save(str(root / "model" / f"{sentence}_{version}.wav"), "WAV")
            sites = {k: {"value": lab, "start": s, "end": e} for k, lab, (s, e) in zip(ids, labels, spans)}
            ap.write_sites(root / "model" / f"{sentence}_{version}.TextGrid", snd.duration, sites)
            # P01 imitates the version heard; P02 always says [e].
            snd, _ = utterance(labels, scale=1.12, f0=200)
            snd.save(str(root / "participants" / f"P01_{sentence}_{version}.wav"), "WAV")
            fixed = [lab if k[0] == "R" else "e" for k, lab in zip(ids, labels)]
            snd, _ = utterance(fixed, scale=1.08, f0=180)
            snd.save(str(root / "participants" / f"P02_{sentence}_{version}.wav"), "WAV")
    return root


def test_end_to_end_formant(corpus, tmp_path):
    ap.analyse(corpus, tmp_path, "formant")
    rows = ap.pd.read_csv(tmp_path / "sites.csv")
    assert len(rows) == 8
    assert (rows["duration"] > 0.1).all()
    imitator, fixed = rows[rows.participant == "P01"], rows[rows.participant == "P02"]
    assert (imitator.value_estimated == imitator.value_heard).all()
    assert (fixed.value_estimated == "e").all()
    assert (corpus / "participants" / "P01_S01_A.auto.TextGrid").exists()
    assert "priming slope" in (tmp_path / "model.txt").read_text()


def test_manual_textgrid_overrides_projection(corpus, tmp_path):
    auto = corpus / "participants" / "P02_S01_A.auto.TextGrid"
    manual = corpus / "participants" / "P02_S01_A.TextGrid"
    sites = ap.read_sites(auto)
    sites["V1"]["value"] = "e"
    ap.write_sites(manual, parselmouth.Sound(str(auto.with_suffix("").with_suffix(".wav"))).duration, sites)
    try:
        ap.analyse(corpus, tmp_path, "formant")
    finally:
        manual.unlink()
    row = ap.pd.read_csv(tmp_path / "sites.csv").query("participant == 'P02' and sentence == 'S01' and heard == 'A'")
    assert row.textgrid.item() == "manual"
    assert row.value_coded.item() == "e"
    assert "ear vs machine: 1/1 agree" in (tmp_path / "model.txt").read_text()


def test_sites_round_trip(tmp_path):
    sites = {"V1": {"value": "e", "start": 0.1, "end": 0.2}, "L1": {"value": "z", "start": 0.2, "end": 0.3}}
    ap.write_sites(tmp_path / "x.TextGrid", 1.0, sites)
    back = ap.read_sites(tmp_path / "x.TextGrid")
    assert back.keys() == sites.keys()
    for k, s in sites.items():
        assert back[k]["value"] == s["value"]
        assert back[k]["start"] == pytest.approx(s["start"])
        assert back[k]["end"] == pytest.approx(s["end"])


def test_dtw_identical_sequences_follow_diagonal():
    x = np.random.default_rng(1).standard_normal((30, 4))
    path, cost = ap.dtw_path(x, x)
    assert cost == pytest.approx(0.0)
    assert (path[:, 0] == path[:, 1]).all()


def test_dtw_stretched_sequence_maps_frames_to_their_copies():
    x = np.random.default_rng(2).standard_normal((20, 4))
    y = np.repeat(x, 2, axis=0)
    path, _ = ap.dtw_path(x, y)
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (19, 39)
    assert all(j // 2 == i for i, j in path)


def test_speech_frames_drops_long_pauses_only():
    times = np.arange(0, 2.0, 0.01)
    level = np.full_like(times, 20.0)
    for lo, hi in ((0.2, 0.6), (0.7, 0.9), (1.4, 1.8)):  # 100 ms pause, then 500 ms pause
        level[(times >= lo) & (times < hi)] = 80.0
    kept = times[ap.speech_frames(times, level)]
    assert kept.min() == pytest.approx(0.2 - ap.EDGE, abs=0.011)
    assert kept.max() == pytest.approx(1.79 + ap.EDGE, abs=0.011)
    assert np.any(np.isclose(kept, 0.65))  # short pause kept
    assert not np.any(np.isclose(kept, 1.15))  # long pause interior dropped
    assert np.any(np.isclose(kept, 0.92))  # margin after speech kept


def test_plausible_reference():
    assert ap.plausible_reference("i", 300, 2300)
    assert not ap.plausible_reference("i", 800, 2300)
    assert not ap.plausible_reference("a", math.nan, 1300)
    assert ap.plausible_reference("o", 450, 900)  # no range for unknown labels


def test_reference_stats_weights_labels_equally():
    toks = [
        ("a", ("S1", "R1"), np.array([700.0, 1300.0])),
        ("i", ("S1", "R2"), np.array([300.0, 2300.0])),
        ("i", ("S2", "R1"), np.array([300.0, 2300.0])),
    ]
    mu, sd = ap.reference_stats(toks)
    assert mu == pytest.approx([500.0, 1800.0])
    assert sd == pytest.approx([200.0, 500.0])
    assert ap.reference_stats(toks[:2]) is None  # fewer than 3 tokens
    assert ap.reference_stats(toks[1:] + toks[1:]) is None  # one label


class FakeRecording:
    """FakeRecording returns formants that agree across tokens only at the true ceiling."""

    def __init__(self, f, true_ceiling, jitter):
        self.f, self.true, self.jitter = f, true_ceiling, jitter

    def vowel(self, _site, ceiling):
        err = 1 + self.jitter * (ceiling - self.true) / 1000
        return self.f[0] * err, self.f[1] * err


def test_choose_ceiling_finds_consistent_setting():
    refs = [(FakeRecording((750, 1300), 5000, j), None, "a", ("S", "R1")) for j in (-0.3, 0.0, 0.3)]
    refs += [(FakeRecording((300, 2300), 5000, j), None, "i", ("S", "R2")) for j in (-0.2, 0.2)]
    assert ap.choose_ceiling(refs) == 5000


def test_contrast_directions_and_place():
    feats = {
        ("S1", "A"): np.array([1.0, 0.0]),
        ("S1", "B"): np.array([-1.0, 0.0]),
        ("S2", "A"): np.array([-1.2, 0.1]),
        ("S2", "B"): np.array([1.2, 0.1]),
    }
    values = {"S1": ("e", "E"), "S2": ("E", "e")}
    talker = {s: {v: (None, {"V1": {"value": values[s][i]}}) for i, v in enumerate("AB")} for s in values}
    dirs = ap.contrast_directions(talker, lambda s, v, k: feats[(s, v)])
    u, size, noise = dirs[("E", "e")]
    assert u == pytest.approx([-1.0, 0.0])
    assert size == pytest.approx(2.2)
    assert noise == pytest.approx(0.1414, abs=1e-3)
    fh, fo = feats[("S1", "A")], feats[("S1", "B")]  # heard [e], other [E]
    assert ap.place(fh, fh, fo, "e", "E", dirs)[0] == pytest.approx(1 / 1.1 * 0.5 + 0.5)
    assert ap.place(fo, fh, fo, "e", "E", dirs)[0] == pytest.approx(0.5 - 1 / 1.1 * 0.5)
    pos, off, uncertain = ap.place(np.array([0.0, 0.5]), fh, fo, "e", "E", dirs)
    assert pos == pytest.approx(0.5) and off == pytest.approx(0.5 / 2.2) and uncertain
    assert math.isnan(ap.place(fh, fh, fo, "o", "O", dirs)[0])


def test_kappa():
    assert ap.kappa(["e", "E", "e"], ["e", "E", "e"]) == pytest.approx(1.0)
    assert ap.kappa(["e", "e", "E", "E"], ["e", "E", "e", "E"]) == pytest.approx(0.0)


def rows_for(participant, imitates, n=6):
    rows = []
    for i in range(n):
        for heard, other in (("A", "B"), ("B", "A")):
            vh, vo = ("e", "E") if heard == "A" else ("E", "e")
            rows.append(
                {
                    "participant": participant,
                    "sentence": f"S{i}",
                    "site": "V1",
                    "heard": heard,
                    "other": other,
                    "value_heard": vh,
                    "value_other": vo,
                    "value_coded": "",
                    "value_estimated": vh if imitates else "e",
                    "position": 0.9 if imitates else 0.5,
                }
            )
    return rows


def test_oriented_codes_against_first_version():
    r = rows_for("P", True, 1)
    assert ap.oriented(r[0])["heard"] == 0.5 and ap.oriented(r[0])["y"] == 1.0
    assert ap.oriented(r[1])["heard"] == -0.5 and ap.oriented(r[1])["y"] == 0.0
    assert ap.oriented(r[1])["pos"] == pytest.approx(0.1)


def test_fit_model_separates_imitators_from_fixed_producers(tmp_path):
    rows = [r for p in ("P1", "P2", "P3") for r in rows_for(p, True)] + rows_for("P4", False)
    text = ap.fit_model(rows, tmp_path)
    b = float(text.split("priming slope   b = ")[1].split()[0])
    assert b > 1
    rate = text.split("heard ref - heard other: ")[1].split()
    assert float(rate[0]) == pytest.approx(0.75)
    assert float(rate[3]) < 0.05
    assert (tmp_path / "model.txt").exists()
