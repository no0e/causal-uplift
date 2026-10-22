"""Download the Criteo uplift dataset (v2.1) and convert it to Parquet.

    python scripts/fetch_criteo.py                 # data/criteo.parquet
    python scripts/fetch_criteo.py --synthetic 200000

13,979,592 rows, about 300 MB gzipped. Criteo AI Lab, Diemert et al., "A
Large Scale Benchmark for Uplift Modeling", AdKDD 2018; distributed under
CC BY-NC-SA 4.0, so it is downloaded here and never committed.

The conversion streams the CSV through DuckDB, so the file is never held in
memory. --synthetic writes a file with the same schema and a known effect
instead, which is what CI runs on.
"""
import argparse
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causal_uplift.data import URL, connect, prepare, summary  # noqa: E402
from causal_uplift.synthetic import write_csv  # noqa: E402


def download(url, target):
    if target.exists():
        print(f"Using {target}")
        return target
    print(f"Downloading {url}")
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=600) as response, \
            open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.rename(target)
    print(f"  {target.stat().st_size / 1e6:.0f} MB")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--synthetic", type=int, default=None,
                        help="Write N synthetic rows instead of downloading.")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        source = write_csv(data / "synthetic.csv", args.synthetic)
        target = data / "synthetic.parquet"
    else:
        source = download(URL, data / "criteo-uplift-v2.1.csv.gz")
        target = data / "criteo.parquet"

    started = time.time()
    con = connect(args.threads)
    rows = prepare(source, target, con)
    print(f"{rows:,} rows -> {target} ({target.stat().st_size / 1e6:.0f} MB) "
          f"in {time.time() - started:.0f}s")
    facts = summary(target, con)
    print(f"  splits {facts['splits']}, treated {facts['treated_share']:.3f}, "
          f"visit ATE {facts['ate']:+.5f} "
          f"[{facts['ate_ci95'][0]:+.5f}, {facts['ate_ci95'][1]:+.5f}], "
          f"max |SMD| {facts['max_abs_smd']:.4f}")


if __name__ == "__main__":
    main()
