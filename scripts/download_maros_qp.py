"""Fetch Maros-Mészáros QPS/SIF benchmark instances into data/qp/.

Adapted from the teammate drop.  The SIF datasets are large benchmark
inputs and are intentionally NOT committed to the repository; this script
regenerates them locally from the public CUTEst/Netlib sources.  If the
public download is unavailable, place the .SIF files manually under
``data/qp/real_benchmarks/maros_meszaros/``.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# Public mirrors for the Maros-Mészáros quadratic programming set.
BASE_URLS = [
    "https://raw.githubusercontent.com/ooitest/SIF-files/main/qpdata/",
    "https://www.cuter.rl.ac.uk//ftp/qpdata/",
]

INSTANCES = ["QAFIRO", "QADLITTLE", "QSC205", "QGROW15", "QPCBOEI1",
             "QSHIP12S"]


def fetch(name: str, out_dir: Path) -> bool:
    target = out_dir / f"{name}.SIF"
    if target.exists():
        print(f"EXISTS {target}")
        return True
    for base in BASE_URLS:
        url = base + f"{name}.SIF"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                target.write_bytes(resp.read())
            print(f"OK    {url} -> {target}")
            return True
        except Exception as exc:  # noqa: BLE001 - best-effort fetch
            print(f"FAIL  {url}: {exc}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parents[1]
                                / "data" / "qp" / "real_benchmarks"
                                / "maros_meszaros"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    got = [name for name in INSTANCES if fetch(name, out_dir)]
    print(f"\nfetched {len(got)}/{len(INSTANCES)} instances -> {out_dir}")


if __name__ == "__main__":
    main()
