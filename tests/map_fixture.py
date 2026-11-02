"""Map chunked site times onto the continuous same-voice audio: model-talker TextGrids + participant ground truth."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyse_priming as ap

c = Path(sys.argv[1])
for sub, dest in (("model", c / "model"), ("participants", c / "gt")):
    dest.mkdir(exist_ok=True)
    for wav in sorted((c / "chunked" / sub).glob("*.wav")):
        src, dst = ap.Recording(wav, None), ap.Recording(c / sub / wav.name, None)
        sites = ap.read_sites(wav.with_suffix(".TextGrid"))
        out, _ = ap.project_sites(src, sites, dst)
        for k in out:
            out[k]["value"] = sites[k]["value"]
        ap.write_sites(dest / wav.with_suffix(".TextGrid").name, dst.duration, out)
