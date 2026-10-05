"""Download the Kaggle datasets used by this package.

    python -m ml.kaggle_fetch            # run from backend/

The token is read from KAGGLE_API_TOKEN (environment or the repo's .env);
it is never written anywhere by this script.  Files land in ``data/raw/``
which is git-ignored.

Datasets, and why each one is (or is not) used:

* ``avikdas567/multi-machine-fusion-detachment-and-fueling-database``
  Real, published (Zenodo) multi-machine table: geometry, B, Ip, power,
  stored energy and H98 for JET, AUG, DIII-D, C-Mod, NSTX, ... .  Small
  (463 rows, ~30 with enough columns for a confinement check), so it is a
  *benchmark*, not a training set.
* ``sergiobuilds/synthetic-fusion-reactor-plasma``
  5000 synthetic discharges with a disruption label.  Synthetic -- used
  only for the demonstrator disruption-risk model.
* ``hark99/multi-machine-disruption-prediction-challenge``
  Re-upload of the ITU/Zindi multi-machine disruption challenge (license
  not stated on Kaggle).  Only the J-TEXT part is fetched (2136 labelled
  HDF5 shots, ~1 GB): it is uncompressed, carries ``meta/IsDisrupt`` and
  ``meta/DownTime`` and 1 kHz diagnostics.  HL-2A (zipped, other tags) and
  C-Mod (the evaluation labels are withheld) are not fetched.

Rejected: ``adebusayoadewunmi/nuclearfusion-data`` (100k rows) -- every
column is statistically independent of every other (|r| < 0.01 against
Ignition and Confinement Time), i.e. random numbers; a model trained on it
would learn nothing real.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
RAW = BACKEND / "data" / "raw"

DATASETS = {
    "multi-machine-fusion-detachment-and-fueling-database":
        "avikdas567/multi-machine-fusion-detachment-and-fueling-database",
    "synthetic-fusion-reactor-plasma":
        "sergiobuilds/synthetic-fusion-reactor-plasma",
}


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _api():
    _load_dotenv()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit("KAGGLE_API_TOKEN is not set (see .env.example)")
    # the client authenticates on import, so import only after the env is set
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


DISRUPTION_REF = "hark99/multi-machine-disruption-prediction-challenge"
JTEXT_PREFIX = "J-TEXT data/J-TEXT data/data/"
JTEXT_DIR = RAW / "jtext"


def fetch_jtext(workers: int = 3, limit: int = 0) -> int:
    """Download the J-TEXT shots; resumable (existing files are skipped)."""
    from concurrent.futures import ThreadPoolExecutor
    api = _api()
    names, tok = [], None
    while True:
        r = api.dataset_list_files(DISRUPTION_REF, page_token=tok,
                                   page_size=200)
        names += [f.name for f in r.files
                  if f.name.startswith(JTEXT_PREFIX)
                  and f.name.endswith(".hdf5")]
        tok = r.next_page_token
        if not tok:
            break
    names = names[:limit] if limit else names
    todo = [n for n in names
            if not (JTEXT_DIR / Path(n).name).is_file()]
    print(f"J-TEXT: {len(names)} shots, {len(todo)} to download")
    JTEXT_DIR.mkdir(parents=True, exist_ok=True)

    def one(n):
        import time
        target = JTEXT_DIR / Path(n).name
        err = None
        for attempt in range(6):
            try:
                api.dataset_download_file(DISRUPTION_REF, n,
                                          path=str(JTEXT_DIR),
                                          force=False, quiet=True)
                time.sleep(0.1)
                return True
            except Exception as exc:          # rate limit / transient
                err = exc
                target.unlink(missing_ok=True)   # never keep a partial file
                time.sleep(1.5 ** attempt)
        print("failed", Path(n).name, type(err).__name__, str(err)[:80])
        return False

    with ThreadPoolExecutor(workers) as pool:
        done = sum(pool.map(one, todo))
    print(f"downloaded {done}/{len(todo)} into {JTEXT_DIR}")
    return done


def fetch(force: bool = False) -> None:
    api = _api()
    for folder, ref in DATASETS.items():
        dest = RAW / folder
        if dest.is_dir() and any(dest.iterdir()) and not force:
            print(f"skip   {ref} (already in {dest})")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"fetch  {ref}")
        api.dataset_download_files(ref, path=str(dest), unzip=True, quiet=True)
    print("done:", RAW)


if __name__ == "__main__":
    if "--jtext" in sys.argv:
        limit = 0
        for i, a in enumerate(sys.argv):
            if a.startswith("--limit="):
                limit = int(a.split("=")[1])
            elif a == "--limit" and i + 1 < len(sys.argv):
                limit = int(sys.argv[i + 1])
        fetch_jtext(limit=limit)
    else:
        fetch(force="--force" in sys.argv)
