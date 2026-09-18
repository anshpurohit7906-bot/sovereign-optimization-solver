"""Reader for the Maros-Mészáros QPS/SIF benchmark format.

The reader supports the sections used by the convex Maros-Mészáros collection:
ROWS, COLUMNS, RHS, RANGES, BOUNDS, QUADOBJ, ENDATA.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import scipy.sparse as sp
from .problem import QPProblem, QPValidationError


def read_qps(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    section = None
    rows, cols = {}, []
    row_types = {}
    rhs = {}
    ranges = {}
    bounds_lo, bounds_up = {}, {}
    q_entries, linear = [], {}
    objective_row = None

    for raw in lines:
        line = raw.rstrip()
        if not line or line.lstrip().startswith("*"):
            continue
        head = line.strip().upper()
        if head in {"ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "QUADOBJ", "ENDATA"}:
            section = head
            if section == "ENDATA":
                break
            continue

        parts = line.split()
        if section == "ROWS":
            if len(parts) >= 2:
                typ, name = parts[0].upper(), parts[1]
                row_types[name] = typ
                if typ == "N":
                    objective_row = name
                else:
                    rows.setdefault(name, len(rows))
        elif section == "COLUMNS":
            if not parts:
                continue
            var = parts[0]
            if var not in cols:
                cols.append(var)
            vals = parts[1:]
            for i in range(0, len(vals) - 1, 2):
                rname, value = vals[i], float(vals[i + 1])
                if rname == objective_row:
                    linear[var] = linear.get(var, 0.0) + value
                elif rname in rows:
                    # Accumulate duplicate entries rather than silently overwrite.
                    key = (rows[rname], cols.index(var))
                    linear[key] = linear.get(key, 0.0) + value
        elif section == "RHS":
            vals = parts[1:]  # first token is RHS vector name
            for i in range(0, len(vals) - 1, 2):
                rhs[vals[i]] = float(vals[i + 1])
        elif section == "RANGES":
            vals = parts[1:]  # first token is RANGES vector name
            for i in range(0, len(vals) - 1, 2):
             ranges[vals[i]] = float(vals[i + 1])
        elif section == "BOUNDS":
            if len(parts) >= 3:
                btyp, var, value = parts[0].upper(), parts[2], None
                if len(parts) >= 4:
                    value = float(parts[3])
                if btyp in {"LO", "LI"}:
                    bounds_lo[var] = value
                elif btyp in {"UP", "UI"}:
                    bounds_up[var] = value
                elif btyp == "FX":
                    bounds_lo[var] = bounds_up[var] = value
                elif btyp == "FR":
                    bounds_lo[var] = -np.inf
                    bounds_up[var] = np.inf
                elif btyp == "MI":
                    bounds_lo[var] = -np.inf
                elif btyp == "PL":
                    bounds_up[var] = np.inf
        elif section == "QUADOBJ":
            if len(parts) >= 3:
                q_entries.append((parts[0], parts[1], float(parts[2])))

    if not cols:
        raise QPValidationError(f"No variables found in {path}")
    index = {v: i for i, v in enumerate(cols)}
    n = len(cols)

    q = np.zeros(n)
    P_rows, P_cols, P_data = [], [], []
    for key, value in linear.items():
        if isinstance(key, str):
            q[index[key]] += value

        # Linear constraints are represented as equalities and <= inequalities.
    A_rows, A_cols, A_data, b_eq = [], [], [], []
    G_rows, G_cols, G_data, h_ineq = [], [], [], []

    for rname, ri in rows.items():
        typ = row_types[rname]
        target = float(rhs.get(rname, 0.0))
        range_value = ranges.get(rname)

        # Collect the sparse coefficients for this row once.
        row_entries = [
            (j, linear[(ri, j)])
            for j in range(n)
            if (ri, j) in linear
        ]

        if typ == "E":
            if range_value is None:
                # Normal equality:
                # a(x) = rhs
                aidx = len(b_eq)
                b_eq.append(target)

                for j, value in row_entries:
                    A_rows.append(aidx)
                    A_cols.append(j)
                    A_data.append(value)

            else:
                # Ranged equality.
                #
                # range >= 0:
                # rhs <= a(x) <= rhs + range
                #
                # range < 0:
                # rhs + range <= a(x) <= rhs

                if range_value >= 0:
                    lower = target
                    upper = target + range_value
                else:
                    lower = target + range_value
                    upper = target

                # a(x) <= upper
                gidx = len(h_ineq)
                h_ineq.append(upper)

                for j, value in row_entries:
                    G_rows.append(gidx)
                    G_cols.append(j)
                    G_data.append(value)

                # a(x) >= lower
                # -a(x) <= -lower
                gidx = len(h_ineq)
                h_ineq.append(-lower)

                for j, value in row_entries:
                    G_rows.append(gidx)
                    G_cols.append(j)
                    G_data.append(-value)

        elif typ == "L":
            # Normal L row:
            # a(x) <= rhs
            #
            # With a range:
            # rhs - |range| <= a(x) <= rhs

            upper = target
            lower = (
                target - abs(range_value)
                if range_value is not None
                else None
            )

            # a(x) <= upper
            gidx = len(h_ineq)
            h_ineq.append(upper)

            for j, value in row_entries:
                G_rows.append(gidx)
                G_cols.append(j)
                G_data.append(value)

            # a(x) >= lower
            # -a(x) <= -lower
            if lower is not None:
                gidx = len(h_ineq)
                h_ineq.append(-lower)

                for j, value in row_entries:
                    G_rows.append(gidx)
                    G_cols.append(j)
                    G_data.append(-value)

        elif typ == "G":
            # Normal G row:
            # a(x) >= rhs
            #
            # With a range:
            # rhs <= a(x) <= rhs + |range|

            lower = target
            upper = (
                target + abs(range_value)
                if range_value is not None
                else None
            )

            # a(x) >= lower
            # -a(x) <= -lower
            gidx = len(h_ineq)
            h_ineq.append(-lower)

            for j, value in row_entries:
                G_rows.append(gidx)
                G_cols.append(j)
                G_data.append(-value)

            # a(x) <= upper
            if upper is not None:
                gidx = len(h_ineq)
                h_ineq.append(upper)

                for j, value in row_entries:
                    G_rows.append(gidx)
                    G_cols.append(j)
                    G_data.append(value)

    for v1, v2, value in q_entries:
        i, j = index[v1], index[v2]
        P_rows.append(i); P_cols.append(j); P_data.append(value)
        if i != j:
            P_rows.append(j); P_cols.append(i); P_data.append(value)
    P = sp.coo_matrix((P_data, (P_rows, P_cols)), shape=(n, n)).tocsr()
    A = (sp.coo_matrix((A_data, (A_rows, A_cols)),
                       shape=(len(b_eq), n)).tocsr()
         if b_eq else None)
    G = (sp.coo_matrix((G_data, (G_rows, G_cols)),
                       shape=(len(h_ineq), n)).tocsr()
         if h_ineq else None)

    lb = np.array([bounds_lo.get(v, 0.0) for v in cols], dtype=float)
    ub = np.array([bounds_up.get(v, np.inf) for v in cols], dtype=float)
    return QPProblem(P, q, G=G, h=np.asarray(h_ineq) if h_ineq else None,
                     A=A, b=np.asarray(b_eq) if b_eq else None,
                     lb=lb, ub=ub, name=Path(path).stem)
