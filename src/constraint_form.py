"""Convert NumericalLP to unified bounded-constraint form: l^c <= A x <= u^c, l^v <= x <= u^v."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from numerical_model import NumericalLP, load_numeric_mps, validate_numeric_lp


@dataclass(frozen=True)
class BoundedConstraintLP:
    """Unified bounded-constraint representation.

    Constraints: constraint_lower <= A x <= constraint_upper
    Variables:   variable_lower <= x <= variable_upper
    """
    name: str
    objective_name: str
    A: np.ndarray
    c: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    row_names: tuple[str, ...]
    var_names: tuple[str, ...]

    @property
    def num_vars(self) -> int:
        return self.c.shape[0]

    @property
    def num_constraints(self) -> int:
        return self.constraint_lower.shape[0]

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(self.A))


def to_bounded_constraint(lp: NumericalLP) -> BoundedConstraintLP:
    """Convert NumericalLP to bounded-constraint form.

    Row type mapping:
      E: lower = upper = b_i
      L: lower = -inf, upper = b_i
      G: lower = b_i, upper = +inf
    """
    m = lp.num_constraints
    n = lp.num_vars

    constraint_lower = np.full(m, -np.inf, dtype=np.float64)
    constraint_upper = np.full(m, np.inf, dtype=np.float64)

    for i, row_type in enumerate(lp.row_types):
        rhs_val = lp.b[i]
        if row_type == "E":
            constraint_lower[i] = rhs_val
            constraint_upper[i] = rhs_val
        elif row_type == "L":
            constraint_lower[i] = -np.inf
            constraint_upper[i] = rhs_val
        elif row_type == "G":
            constraint_lower[i] = rhs_val
            constraint_upper[i] = np.inf
        else:
            raise ValueError(f"Unknown row type: {row_type}")

    return BoundedConstraintLP(
        name=lp.name,
        objective_name=lp.objective_name,
        A=lp.A.copy(),
        c=lp.c.copy(),
        constraint_lower=constraint_lower,
        constraint_upper=constraint_upper,
        variable_lower=lp.lower_bounds.copy(),
        variable_upper=lp.upper_bounds.copy(),
        row_names=lp.row_names,
        var_names=lp.var_names,
    )


def validate_bounded_constraint(lp: BoundedConstraintLP) -> None:
    """Validate the bounded-constraint representation."""
    m, n = lp.A.shape

    if lp.c.shape != (n,):
        raise ValueError(f"c shape mismatch: {lp.c.shape} != ({n},)")
    if lp.constraint_lower.shape != (m,) or lp.constraint_upper.shape != (m,):
        raise ValueError("constraint bounds shape mismatch")
    if lp.variable_lower.shape != (n,) or lp.variable_upper.shape != (n,):
        raise ValueError("variable bounds shape mismatch")
    if len(lp.row_names) != m or len(lp.var_names) != n:
        raise ValueError("metadata length mismatch")

    # Check constraint bounds consistency
    if np.any(lp.constraint_lower > lp.constraint_upper):
        raise ValueError("constraint lower bound exceeds upper bound")
    if np.any(lp.variable_lower > lp.variable_upper):
        raise ValueError("variable lower bound exceeds upper bound")

    # Check that all finite constraint bounds are consistent with original row types
    finite_lower = np.isfinite(lp.constraint_lower)
    finite_upper = np.isfinite(lp.constraint_upper)
    if not np.allclose(lp.constraint_lower[finite_lower & finite_upper],
                       lp.constraint_upper[finite_lower & finite_upper]):
        raise ValueError("equality constraints must have equal lower/upper bounds")


def _self_test() -> None:
    """Self-test with E, L, G rows."""
    A = np.array([
        [1.0, 2.0],   # E row
        [3.0, 4.0],   # L row
        [5.0, 6.0],   # G row
    ], dtype=np.float64)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    c = np.array([1.0, 2.0], dtype=np.float64)
    variable_lower = np.array([0.0, 0.0], dtype=np.float64)
    variable_upper = np.array([np.inf, np.inf], dtype=np.float64)
    row_types = ("E", "L", "G")
    row_names = ("eq_row", "le_row", "ge_row")
    var_names = ("x1", "x2")

    lp = NumericalLP(
        name="TEST",
        objective_name="OBJ",
        A=A,
        b=b,
        c=c,
        lower_bounds=variable_lower,
        upper_bounds=variable_upper,
        row_types=row_types,
        var_names=var_names,
        row_names=row_names,
    )

    bounded = to_bounded_constraint(lp)
    validate_bounded_constraint(bounded)

    print("=== Self-Test: Bounded Constraint Conversion ===")
    print(f"Problem: {bounded.name}")
    print(f"A shape: {bounded.A.shape}")
    print(f"c shape: {bounded.c.shape}")
    print(f"Constraint lower: {bounded.constraint_lower}")
    print(f"Constraint upper: {bounded.constraint_upper}")
    print(f"Variable lower: {bounded.variable_lower}")
    print(f"Variable upper: {bounded.variable_upper}")
    print(f"Row names: {bounded.row_names}")
    print(f"Var names: {bounded.var_names}")

    # Verify mapping
    assert np.isclose(bounded.constraint_lower[0], 10.0) and np.isclose(bounded.constraint_upper[0], 10.0), "E row failed"
    assert np.isneginf(bounded.constraint_lower[1]) and np.isclose(bounded.constraint_upper[1], 20.0), "L row failed"
    assert np.isclose(bounded.constraint_lower[2], 30.0) and np.isposinf(bounded.constraint_upper[2]), "G row failed"
    assert np.allclose(bounded.variable_lower, [0.0, 0.0]), "Variable lower failed"
    assert np.all(np.isposinf(bounded.variable_upper)), "Variable upper failed"

    print("\nSelf-test: PASS")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert MPS to bounded-constraint LP.")
    parser.add_argument("mps_file", nargs="?")
    parser.add_argument("--self-test", action="store_true", help="Run self-test only")
    args = parser.parse_args()

    if args.self_test or args.mps_file is None:
        _self_test()
        return

    from pathlib import Path
    lp = load_numeric_mps(args.mps_file)
    bounded = to_bounded_constraint(lp)
    validate_bounded_constraint(bounded)

    print(f"Problem: {bounded.name}")
    print(f"Objective: {bounded.objective_name}")
    print(f"Variables: {bounded.num_vars}")
    print(f"Constraints: {bounded.num_constraints}")
    print(f"NNZ: {bounded.nnz}")
    print()
    print("Constraint bounds (l^c <= Ax <= u^c):")
    for i, (name, lo, up) in enumerate(zip(bounded.row_names, bounded.constraint_lower, bounded.constraint_upper)):
        lo_str = f"{lo:.6f}" if np.isfinite(lo) else "-inf"
        up_str = f"{up:.6f}" if np.isfinite(up) else "+inf"
        print(f"  {i}: {name:12s}  [{lo_str}, {up_str}]")
    print()
    print("Variable bounds (l^v <= x <= u^v):")
    for i, (name, lo, up) in enumerate(zip(bounded.var_names, bounded.variable_lower, bounded.variable_upper)):
        lo_str = f"{lo:.6f}" if np.isfinite(lo) else "-inf"
        up_str = f"{up:.6f}" if np.isfinite(up) else "+inf"
        print(f"  {i}: {name:12s}  [{lo_str}, {up_str}]")


if __name__ == "__main__":
    main()