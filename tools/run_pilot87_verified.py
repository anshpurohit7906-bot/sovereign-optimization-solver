"""Orchestrate the validated PILOT87 crossover pipeline end-to-end.

Sole responsibility: run the four validated stages in order and make the
artifact flow automatic.  It implements NO solver or algorithmic logic - it
only invokes the existing, validated stage scripts in sequence:

    data/pilot87.mps
  -> artifacts/pilot87/p87_prepared.npz            (preparation)
  -> artifacts/pilot87/p87_phase2_v2_final.npz     (Phase II)
  -> artifacts/pilot87/p87_strict_polished.npz     (strict polish)
  -> artifacts/pilot87/p87_strict_certificate.txt  (polish stdout)
  -> artifacts/pilot87/p87_certificate.txt         (certification)

All generated artifacts live under artifacts/pilot87/ (canonical location).
There is no scratch/ handoff: each stage reads exactly what the previous
stage wrote, directly from the canonical artifact directory.  The strict-polish
stage's stdout is captured into the strict certificate
(p87_strict_certificate.txt), and the certification stage's report is captured
into its own certificate (p87_certificate.txt).  Both certificate files are
verified to exist and be non-empty before the pipeline is considered complete.

Usage:
    OPENBLAS_NUM_THREADS=1 python tools/run_pilot87_verified.py
"""
from __future__ import annotations

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

MPS_PATH = os.path.join(_ROOT, "data", "pilot87.mps")
ARTIFACT_DIR = os.path.join(_ROOT, "artifacts", "pilot87")

# Each stage runs its own script as a subprocess, inheriting this env.
STAGE_ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1")

PREPARE = os.path.join(_ROOT, "experiment", "crossover", "p87_prepare.py")
PHASE2 = os.path.join(_ROOT, "experiment", "crossover", "p87_phase2_v2.py")
POLISH = os.path.join(_ROOT, "tools", "certification", "p87_strict_polish.py")
CERTIFY = os.path.join(_ROOT, "tools", "certification", "p87_certify.py")

# Expected canonical artifacts produced at each stage boundary.
PREPARED_NPZ = os.path.join(ARTIFACT_DIR, "p87_prepared.npz")
PREPARED_A_NPZ = os.path.join(ARTIFACT_DIR, "p87_prepared_A.npz")
PHASE2_FINAL_NPZ = os.path.join(ARTIFACT_DIR, "p87_phase2_v2_final.npz")
POLISHED_NPZ = os.path.join(ARTIFACT_DIR, "p87_strict_polished.npz")
# The strict certificate (p87_strict_certificate.txt) is produced by capturing
# the strict-polish script's stdout; the certification report is captured into
# its own p87_certificate.txt.  Both are verified to exist and be non-empty.
STRICT_CERT_FILE = os.path.join(ARTIFACT_DIR, "p87_strict_certificate.txt")
CERT_FILE = os.path.join(ARTIFACT_DIR, "p87_certificate.txt")


def _run_stage(label: str, script: str, *args: str) -> int:
    """Run one validated stage as a subprocess and propagate failures."""
    print(f"\n===== STAGE: {label} =====", flush=True)
    cmd = [sys.executable, "-u", script, *args]
    rc = subprocess.call(cmd, env=STAGE_ENV)
    if rc != 0:
        print(f"\n[FAILED] Stage '{label}' exited with code {rc}. "
              f"Pipeline aborted.", flush=True)
        sys.exit(rc)
    print(f"[OK] Stage '{label}' completed (exit 0).", flush=True)
    return rc


def main() -> int:
    if not os.path.isfile(MPS_PATH):
        print(f"ERROR: required data file missing: {MPS_PATH}", flush=True)
        return 1

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ---- 1. Preparation (RRQR + Phase I -> prepared artifacts) ----
    _run_stage("preparation", PREPARE)
    for _f in (PREPARED_NPZ, PREPARED_A_NPZ):
        if not os.path.isfile(_f):
            print(f"ERROR: preparation did not produce expected artifact: {_f}",
                  flush=True)
            return 1

    # ---- 2. Phase II (Devex Revised Simplex -> terminal basis) ----
    _run_stage("phase II", PHASE2)
    if not os.path.isfile(PHASE2_FINAL_NPZ):
        print(f"ERROR: Phase II did not produce expected artifact: "
              f"{PHASE2_FINAL_NPZ}", flush=True)
        return 1

    # ---- 3. Strict polish (-> polished solution + strict certificate) ----
    print(f"\n===== STAGE: strict polish =====", flush=True)
    polish_cmd = [sys.executable, "-u", POLISH]
    with open(STRICT_CERT_FILE, "w", encoding="utf-8") as _strict_out:
        proc = subprocess.Popen(
            polish_cmd, env=STAGE_ENV,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            _strict_out.write(line)
        rc = proc.wait()
    if rc != 0:
        print(f"\n[FAILED] Stage 'strict polish' exited with code {rc}. "
              f"Pipeline aborted.", flush=True)
        sys.exit(rc)
    print(f"[OK] Stage 'strict polish' completed (exit 0).", flush=True)
    if not os.path.isfile(POLISHED_NPZ):
        print(f"ERROR: strict polish did not produce expected artifact: "
              f"{POLISHED_NPZ}", flush=True)
        return 1
    if not os.path.isfile(STRICT_CERT_FILE):
        print(f"ERROR: strict polish did not produce the strict certificate: "
              f"{STRICT_CERT_FILE}", flush=True)
        return 1
    if os.path.getsize(STRICT_CERT_FILE) == 0:
        print(f"ERROR: strict certificate file is empty: {STRICT_CERT_FILE}",
              flush=True)
        return 1

    # ---- 4. Certification (captures printed report -> certificate file) ----
    print(f"\n===== STAGE: certification =====", flush=True)
    cert_cmd = [sys.executable, "-u", CERTIFY]
    with open(CERT_FILE, "w", encoding="utf-8") as _cert_out:
        proc = subprocess.Popen(
            cert_cmd, env=STAGE_ENV,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            # Mirror to console and into the certificate file simultaneously.
            sys.stdout.write(line)
            sys.stdout.flush()
            _cert_out.write(line)
        rc = proc.wait()
    if rc != 0:
        print(f"\n[FAILED] Stage 'certification' exited with code {rc}. "
              f"Pipeline aborted.", flush=True)
        sys.exit(rc)
    print(f"[OK] Stage 'certification' completed (exit 0).", flush=True)
    if not os.path.isfile(CERT_FILE):
        print(f"ERROR: certification did not produce expected artifact: "
              f"{CERT_FILE}", flush=True)
        return 1
    # Empty certificate file means no report was captured — treat as missing.
    if os.path.getsize(CERT_FILE) == 0:
        print(f"ERROR: certificate file is empty: {CERT_FILE}", flush=True)
        return 1

    print(f"\nCertificate:       {CERT_FILE}", flush=True)
    print(f"Strict certificate:{STRICT_CERT_FILE}", flush=True)
    print(f"Polished artifact: {POLISHED_NPZ}", flush=True)
    print("Pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
