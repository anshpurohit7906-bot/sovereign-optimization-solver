"""Convert the validated MPS LPModel into NumPy solver-ready arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mps_parser import LPModel, MPSParser


class NumericalModelError(ValueError):
    """Raised when an LPModel cannot be converted safely."""


@dataclass(frozen=True)
class NumericalLP:
    """Dense numerical representation used for the first CPU experiments."""

    name: str
    objective_name: str
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    row_types: tuple[str, ...]
    var_names: tuple[str, ...]
    row_names: tuple[str, ...]

    @property
    def num_vars(self) -> int:
        return self.c.shape[0]

    @property
    def num_constraints(self) -> int:
        return self.b.shape[0]

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(self.A))


def to_numeric(model: LPModel) -> NumericalLP:
    """Convert a parsed LPModel to dense NumPy arrays.

    The parser stores sparse coefficients as (variable_index, row_index) -> value.
    The dense matrix A is conventionally indexed as A[row, column].
    """
    m = model.num_constraints()
    n = model.num_vars()

    if len(model.rhs) != m:
        raise NumericalModelError(f"RHS length {len(model.rhs)} != constraints {m}")
    if len(model.obj) != n:
        raise NumericalModelError(f"objective length {len(model.obj)} != variables {n}")
    if len(model.bounds_lb) != n or len(model.bounds_ub) != n:
        raise NumericalModelError("bound arrays do not match variable count")
    if len(model.row_types) != m:
        raise NumericalModelError(f"row_types length {len(model.row_types)} != constraints {m}")

    A = np.zeros((m, n), dtype=np.float64)
    for (col_idx, row_idx), value in model.coeffs.items():
        if not (0 <= col_idx < n):
            raise NumericalModelError(f"column index out of range: {col_idx}")
        if not (0 <= row_idx < m):
            raise NumericalModelError(f"row index out of range: {row_idx}")
        A[row_idx, col_idx] = value

    b = np.asarray(model.rhs, dtype=np.float64)
    c = np.asarray(model.obj, dtype=np.float64)
    lower = np.asarray(model.bounds_lb, dtype=np.float64)
    upper = np.asarray(model.bounds_ub, dtype=np.float64)

    if not np.all(np.isfinite(c)):
        raise NumericalModelError("objective contains non-finite values")
    if not np.all(np.isfinite(b)):
        raise NumericalModelError("RHS contains non-finite values")
    if np.any(lower > upper):
        raise NumericalModelError("at least one lower bound exceeds its upper bound")

    return NumericalLP(
        name=model.name,
        objective_name=model.objective_name,
        A=A,
        b=b,
        c=c,
        lower_bounds=lower,
        upper_bounds=upper,
        row_types=tuple(model.row_types),
        var_names=tuple(model.var_names),
        row_names=tuple(model.row_names),
    )


def validate_numeric_lp(lp: NumericalLP, *, expected: Optional[dict] = None) -> None:
    """Run structural checks; optionally enforce known benchmark expectations."""
    m, n = lp.A.shape

    if lp.b.shape != (m,):
        raise NumericalModelError(f"A/b shape mismatch: A={lp.A.shape}, b={lp.b.shape}")
    if lp.c.shape != (n,):
        raise NumericalModelError(f"A/c shape mismatch: A={lp.A.shape}, c={lp.c.shape}")
    if lp.lower_bounds.shape != (n,) or lp.upper_bounds.shape != (n,):
        raise NumericalModelError("bound shape mismatch")
    if len(lp.row_types) != m or len(lp.row_names) != m:
        raise NumericalModelError("row metadata length mismatch")
    if len(lp.var_names) != n:
        raise NumericalModelError("variable metadata length mismatch")

    type_counts = {t: lp.row_types.count(t) for t in ("E", "L", "G")}
    if any(t not in ("E", "L", "G") for t in lp.row_types):
        raise NumericalModelError(f"unexpected row type(s): {lp.row_types}")

    if expected:
        if "name" in expected and lp.name != expected["name"]:
            raise NumericalModelError(f"name mismatch: {lp.name!r} != {expected['name']!r}")
        if "objective_name" in expected and lp.objective_name != expected["objective_name"]:
            raise NumericalModelError(
                f"objective mismatch: {lp.objective_name!r} != {expected['objective_name']!r}"
            )
        if "num_vars" in expected and n != expected["num_vars"]:
            raise NumericalModelError(f"variable count mismatch: {n} != {expected['num_vars']}")
        if "num_constraints" in expected and m != expected["num_constraints"]:
            raise NumericalModelError(
                f"constraint count mismatch: {m} != {expected['num_constraints']}"
            )
        if "nnz" in expected and lp.nnz != expected["nnz"]:
            raise NumericalModelError(f"NNZ mismatch: {lp.nnz} != {expected['nnz']}")
        for t, expected_count in expected.get("row_types", {}).items():
            if type_counts.get(t, 0) != expected_count:
                raise NumericalModelError(
                    f"row type count mismatch for {t}: {type_counts.get(t, 0)} != {expected_count}"
                )
        if expected.get("default_nonnegative_bounds"):
            if not np.all(lp.lower_bounds == 0.0):
                raise NumericalModelError("expected all lower bounds to be 0")
            if not np.all(np.isposinf(lp.upper_bounds)):
                raise NumericalModelError("expected all upper bounds to be +inf")


def load_numeric_mps(path: str | Path) -> NumericalLP:
    """Parse an MPS file and convert it to a numerical LP."""
    model = MPSParser().parse_file(str(path))
    return to_numeric(model)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert MPS to dense numerical LP arrays.")
    parser.add_argument("mps_file")
    args = parser.parse_args()

    lp = load_numeric_mps(args.mps_file)
    validate_numeric_lp(
        lp,
        expected={
            "name": "AFIRO",
            "objective_name": "COST",
            "num_vars": 32,
            "num_constraints": 27,
            "nnz": 83,
            "row_types": {"E": 8, "L": 19, "G": 0},
            "default_nonnegative_bounds": True,
        },
    )

    print(f"Problem name       : {lp.name}")
    print(f"Objective row      : {lp.objective_name}")
    print(f"A shape            : {lp.A.shape}")
    print(f"b shape            : {lp.b.shape}")
    print(f"c shape            : {lp.c.shape}")
    print(f"lower bounds shape : {lp.lower_bounds.shape}")
    print(f"upper bounds shape : {lp.upper_bounds.shape}")
    print(f"NNZ(A)             : {lp.nnz}")
    print("Row types          : E=8, L=19, G=0")
    print("\n--- Sample nonzero entries A[row, col] ---")
    shown = 0
    for r, c in zip(*np.nonzero(lp.A)):
        print(f"  A[{r},{c}] ({lp.row_names[r]}, {lp.var_names[c]}) = {lp.A[r,c]:.6f}")
        shown += 1
        if shown == 8:
            break

    print("\n--- Sample objective coefficients ---")
    for i in np.flatnonzero(lp.c)[:8]:
        print(f"  c[{i}] ({lp.var_names[i]}) = {lp.c[i]:.6f}")

    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
