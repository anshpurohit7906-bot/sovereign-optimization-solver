"""Download the selected Maros-Mészáros QP benchmark files."""
from __future__ import annotations
import argparse
from pathlib import Path
from urllib.request import urlopen

BASE = "https://raw.githubusercontent.com/optimizers/maros-meszaros-mirror/master"
DEFAULT = ["QAFIRO", "QADLITTL", "QSC205", "QGROW15", "QPCBOEI1", "QSHIP12S"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/qp/real_benchmarks/maros_meszaros")
    ap.add_argument("--problems", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name in args.problems:
        url = f"{BASE}/{name}.SIF"
        target = out / f"{name}.SIF"
        print(f"Downloading {name}...")
        with urlopen(url, timeout=60) as src, target.open("wb") as dst:
            dst.write(src.read())
        print(f"  -> {target}")
if __name__ == "__main__":
    main()
