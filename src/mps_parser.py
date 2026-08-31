"""
MPS Parser for LP problems.
Produces a structured LPModel suitable for later conversion to sparse matrices.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


class MPSParseError(Exception):
    """Raised when the MPS file is malformed or inconsistent."""
    pass


@dataclass
class LPModel:
    """
    Immutable internal representation of a linear program parsed from MPS.
    Uses integer indices for all arrays to allow direct sparse-matrix construction.
    """
    name: str = ""
    objective_name: str = ""

    # Ordered name lists for stable indexing
    var_names: List[str] = field(default_factory=list)
    row_names: List[str] = field(default_factory=list)   # constraints only (no objective)

    # Bi-directional name <-> index mappings
    var_index: Dict[str, int] = field(default_factory=dict)
    row_index: Dict[str, int] = field(default_factory=dict)

    # Per-constraint data (indexed by row_index)
    row_types: List[str] = field(default_factory=list)   # 'E', 'L', 'G'
    rhs: List[float] = field(default_factory=list)

    # Per-variable data (indexed by var_index)
    obj: List[float] = field(default_factory=list)       # objective coefficients
    bounds_lb: List[float] = field(default_factory=list) # lower bounds
    bounds_ub: List[float] = field(default_factory=list) # upper bounds

    # Sparse coefficients: (var_idx, row_idx) -> value
    # Suitable for CSR/CSC conversion later.
    coeffs: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def num_vars(self) -> int:
        return len(self.var_names)

    def num_constraints(self) -> int:
        return len(self.row_names)

    def num_nonzeros(self) -> int:
        return len(self.coeffs)

    def count_row_types(self) -> Dict[str, int]:
        counts = {"E": 0, "L": 0, "G": 0}
        for t in self.row_types:
            if t in counts:
                counts[t] += 1
        return counts


class MPSParser:
    """
    Parses an MPS-format LP file and builds an LPModel.

    Supported sections: NAME, ROWS, COLUMNS, RHS, BOUNDS, ENDATA.
    Not supported in this initial version: RANGES, SOS, QUADOBJ, INDICATORS.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.name: str = ""

        # Accumulators during parsing
        self._rows: Dict[str, str] = {}          # row_name -> type ('E','L','G','N')
        self._objective_name: str = ""
        self._cols: Dict[str, None] = {}          # discovered column names, insertion ordered
        self._coeffs: Dict[Tuple[str, str], float] = {}  # (col,row) -> accumulated value
        self._obj_coeffs: Dict[str, float] = {}  # col -> objective coeff
        self._rhs: Dict[str, float] = {}         # row_name -> rhs value
        self._bounds: Dict[str, Tuple[float, float]] = {}  # col -> (lb, ub)

        # State
        self._section: str = ""
        self._line_num: int = 0
        self._rhs_vector_name: str = ""          # first RHS vector name encountered

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize_mps_line(line: str) -> List[str]:
        """
        Tokenize an MPS data line using fixed-format field positions,
        with a fallback to whitespace splitting if the line is very short.

        Standard MPS field layout (1-based columns):
          Field 1:  1-2   (indicator, usually empty for data)
          Field 2:  5-12  (name)
          Field 3:  15-22 (name)
          Field 4:  25-36 (numeric value)
          Field 5:  40-47 (name)
          Field 6:  50-61 (numeric value)
        """
        # Use 0-based indexing internally
        fields = []

        def grab(start_1based: int, end_1based: int) -> str:
            s = start_1based - 1
            e = end_1based
            if s >= len(line):
                return ""
            return line[s:e].strip()

        # Try fixed-format extraction
        f2 = grab(5, 12)
        f3 = grab(15, 22)
        f4 = grab(25, 36)
        f5 = grab(40, 47)
        f6 = grab(50, 61)

        # If the line is too short for fixed format, fall back to whitespace split
        if not f2 and not f3:
            parts = line.strip().split()
            return parts

        # Build field list, excluding empty trailing fields
        if f2:
            fields.append(f2)
        if f3:
            fields.append(f3)
        if f4:
            fields.append(f4)
        if f5:
            fields.append(f5)
        if f6:
            fields.append(f6)

        return fields

    # ------------------------------------------------------------------
    # Section handlers
    # ------------------------------------------------------------------
    def _handle_name(self, line: str):
        """NAME section: first non-header token is the problem name."""
        parts = line.strip().split()
        if len(parts) >= 2:
            self.name = parts[1]
        else:
            self.name = ""

    def _handle_rows(self, fields: List[str]):
        """ROWS section: each line is 'type row_name'."""
        if len(fields) < 2:
            raise MPSParseError(f"Line {self._line_num}: ROWS entry too short: {fields}")
        row_type = fields[0].upper()
        row_name = fields[1]

        if row_type not in ("E", "L", "G", "N"):
            raise MPSParseError(f"Line {self._line_num}: Unknown row type '{row_type}' for row '{row_name}'")

        if row_name in self._rows:
            raise MPSParseError(f"Line {self._line_num}: Duplicate row name '{row_name}'")

        self._rows[row_name] = row_type

        # Track the first N row as the objective
        if row_type == "N":
            if not self._objective_name:
                self._objective_name = row_name

    def _handle_columns(self, fields: List[str]):
        """
        COLUMNS section: each line is:
          col_name  row1_name  value1  [row2_name  value2]
        The objective row entries are extracted separately.
        """
        # Integer markers appear as ordinary COLUMNS data (e.g.
        # "    MARKER  'MARKER'  'INTORG'"), not as a section header, so
        # they must be caught here rather than in the header-detection loop.
        if "'MARKER'" in fields and ("'INTORG'" in fields or "'INTEND'" in fields):
            raise MPSParseError(
                f"Line {self._line_num}: Integer variable markers "
                "(MARKER/INTORG/INTEND) are not implemented by this parser."
            )

        if len(fields) < 3:
            raise MPSParseError(f"Line {self._line_num}: COLUMNS entry too short: {fields}")

        col_name = fields[0]
        self._cols.setdefault(col_name, None)

        # Process up to two (row, value) pairs per line
        pairs = []
        i = 1
        while i + 1 < len(fields):
            row_name = fields[i]
            value_str = fields[i + 1]
            try:
                value = float(value_str)
            except ValueError as exc:
                raise MPSParseError(
                    f"Line {self._line_num}: Invalid numeric value '{value_str}' for ({col_name}, {row_name})"
                ) from exc
            pairs.append((row_name, value))
            i += 2

        for row_name, value in pairs:
            if row_name not in self._rows:
                raise MPSParseError(
                    f"Line {self._line_num}: Row '{row_name}' referenced in COLUMNS but not declared in ROWS"
                )

            if row_name == self._objective_name:
                # Objective coefficient
                self._obj_coeffs[col_name] = self._obj_coeffs.get(col_name, 0.0) + value
            else:
                # Constraint coefficient
                key = (col_name, row_name)
                self._coeffs[key] = self._coeffs.get(key, 0.0) + value

    def _handle_rhs(self, fields: List[str], line: str):
        """
        RHS section: each line is
          rhs_name  row1_name  value1  [row2_name  value2]
        where the RHS vector name is OPTIONAL: per the MPS fixed-format
        layout, the vector name occupies field 2 (columns 5-12) and the
        first row name starts in field 3 (columns 15-22).  A data line
        with empty field 2 therefore omits the vector name and its
        tokens are (row, value) pairs only.
        We use the first RHS vector name encountered and ignore subsequent
        vectors (standard behavior for single-vector MPS files).
        """
        # Field 2 (1-based columns 5-12) is non-empty iff a vector name is
        # present.  ``line`` retains its leading whitespace (only trailing
        # newlines are stripped), so a plain slice reads the fixed field.
        has_vector_name = bool(line[4:12].strip())

        i = 0
        if has_vector_name:
            if len(fields) < 3:
                raise MPSParseError(f"Line {self._line_num}: RHS entry too short: {fields}")
            rhs_name = fields[0]
            if not self._rhs_vector_name:
                self._rhs_vector_name = rhs_name

            # Only process entries from the first RHS vector
            if rhs_name != self._rhs_vector_name:
                return
            i = 1
        elif len(fields) < 2:
            raise MPSParseError(f"Line {self._line_num}: RHS entry too short: {fields}")

        # Process up to two (row, value) pairs
        while i + 1 < len(fields):
            row_name = fields[i]
            value_str = fields[i + 1]
            try:
                value = float(value_str)
            except ValueError as exc:
                raise MPSParseError(
                    f"Line {self._line_num}: Invalid numeric value '{value_str}' in RHS for row '{row_name}'"
                ) from exc

            if row_name not in self._rows:
                raise MPSParseError(
                    f"Line {self._line_num}: Row '{row_name}' referenced in RHS but not declared in ROWS"
                )

            # Don't store RHS for the objective row (it has no RHS meaning)
            if self._rows.get(row_name) != "N":
                self._rhs[row_name] = self._rhs.get(row_name, 0.0) + value

            i += 2

    def _handle_bounds(self, fields: List[str]):
        """
        BOUNDS section: each line is:
          bound_type  bound_name  col_name  [value]
        Supported types: LO, UP, FX, FR, MI, PL.
        """
        if len(fields) < 3:
            raise MPSParseError(f"Line {self._line_num}: BOUNDS entry too short: {fields}")

        bound_type = fields[0].upper()
        # fields[1] is the bound set name (ignored for single-vector files)
        col_name = fields[2]
        value = 0.0
        if len(fields) >= 4:
            try:
                value = float(fields[3])
            except ValueError as exc:
                raise MPSParseError(
                    f"Line {self._line_num}: Invalid numeric value in BOUNDS for '{col_name}'"
                ) from exc

        if col_name not in self._cols:
            raise MPSParseError(
                f"Line {self._line_num}: Column '{col_name}' in BOUNDS not found in COLUMNS"
            )

        # Get current bounds, default [0, +inf]
        lb, ub = self._bounds.get(col_name, (0.0, float("inf")))

        if bound_type == "LO":
            lb = value
        elif bound_type == "UP":
            ub = value
        elif bound_type == "FX":
            lb = ub = value
        elif bound_type == "FR":
            lb = float("-inf")
            ub = float("inf")
        elif bound_type == "MI":
            lb = float("-inf")
        elif bound_type == "PL":
            ub = float("inf")
        else:
            raise MPSParseError(
                f"Line {self._line_num}: Unsupported bound type '{bound_type}' for column '{col_name}'"
            )

        self._bounds[col_name] = (lb, ub)

    # ------------------------------------------------------------------
    # Main parse loop
    # ------------------------------------------------------------------
    def parse_file(self, filepath: str) -> LPModel:
        """Parse an MPS file and return an LPModel."""
        self._reset()

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for raw_line in lines:
            self._line_num += 1
            line = raw_line.rstrip("\n\r")

            # Skip comments and blank lines
            if not line.strip() or line.strip().startswith("*"):
                continue

            # Check for section headers. NAME may be written as
            # "NAME          AFIRO", so inspect the first token rather than
            # requiring the entire line to equal the header.
            stripped = line.strip()
            first_token = stripped.split(None, 1)[0].upper()
            # Section headers are unindented in conventional MPS.
            # Indented records such as "    RHS ROW00001 200." are data,
            # not new RHS section headers.
            is_supported_header = not line[0].isspace() and first_token in (
                "NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"
            )
            # RANGES/SOS/QUADOBJ/INDICATORS are recognized but not yet
            # implemented: reject explicitly at the section header rather
            # than silently ignoring the section's body.
            is_unsupported_header = not line[0].isspace() and first_token in (
                "RANGES", "SOS", "QUADOBJ", "INDICATORS"
            )
            if is_unsupported_header:
                raise MPSParseError(
                    f"Line {self._line_num}: Unsupported MPS section "
                    f"'{first_token}' is not implemented by this parser."
                )
            if is_supported_header:
                self._section = first_token
                if self._section == "NAME":
                    self._handle_name(line)
                continue

            # Tokenize according to section grammar. AFIRO uses conventional
            # whitespace-separated MPS records, and section-specific parsing is
            # safer than applying one fixed-column tokenizer everywhere.
            fields = line.strip().split()
            if not fields:
                continue

            if self._section == "ROWS":
                self._handle_rows(fields)
            elif self._section == "COLUMNS":
                self._handle_columns(fields)
            elif self._section == "RHS":
                # Pass the raw line: the RHS vector name is optional and its
                # presence is decided from fixed-format field 2 (cols 5-12).
                self._handle_rhs(fields, line)
            elif self._section == "BOUNDS":
                self._handle_bounds(fields)
            else:
                raise MPSParseError(
                    f"Line {self._line_num}: Data found outside of a recognized section: {line.strip()}"
                )

        if self._section != "ENDATA":
            raise MPSParseError("MPS file terminated without ENDATA marker.")

        return self._build_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_model(self) -> LPModel:
        """Validate accumulators and construct the immutable LPModel."""
        if not self._objective_name:
            raise MPSParseError("No objective row (type 'N') found in ROWS section.")

        model = LPModel()
        model.name = self.name
        model.objective_name = self._objective_name

        # Build ordered variable list (preserve first-appearance order from COLUMNS)
        model.var_names = list(self._cols.keys())
        for idx, name in enumerate(model.var_names):
            model.var_index[name] = idx

        # Build ordered constraint row list (preserve ROWS order, skip objective)
        model.row_names = [r for r, t in self._rows.items() if t != "N"]
        for idx, name in enumerate(model.row_names):
            model.row_index[name] = idx

        n_vars = len(model.var_names)
        n_rows = len(model.row_names)

        # Initialize arrays
        model.row_types = [self._rows[r] for r in model.row_names]
        model.rhs = [self._rhs.get(r, 0.0) for r in model.row_names]
        model.obj = [self._obj_coeffs.get(v, 0.0) for v in model.var_names]

        # Bounds: default [0, +inf] unless overridden
        for v in model.var_names:
            if v in self._bounds:
                lb, ub = self._bounds[v]
            else:
                lb, ub = 0.0, float("inf")
            model.bounds_lb.append(lb)
            model.bounds_ub.append(ub)

        # Coefficients: convert names to integer indices
        for (col_name, row_name), val in self._coeffs.items():
            if col_name not in model.var_index:
                raise MPSParseError(
                    f"Column '{col_name}' has coefficients but was not declared properly."
                )
            if row_name not in model.row_index:
                raise MPSParseError(
                    f"Row '{row_name}' has coefficients but is not a valid constraint row."
                )
            cidx = model.var_index[col_name]
            ridx = model.row_index[row_name]
            model.coeffs[(cidx, ridx)] = val

        return model


# ----------------------------------------------------------------------
# CLI / demo
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python mps_parser.py <file.mps>")
        sys.exit(1)

    filepath = sys.argv[1]
    parser = MPSParser()

    try:
        model = parser.parse_file(filepath)
    except MPSParseError as exc:
        print(f"Parse error: {exc}")
        sys.exit(1)

    # Summary
    print(f"Problem name       : {model.name}")
    print(f"Objective row      : {model.objective_name}")
    print(f"Variables          : {model.num_vars()}")
    print(f"Constraints        : {model.num_constraints()}")
    print(f"Nonzero coeffs     : {model.num_nonzeros()}")

    type_counts = model.count_row_types()
    print(f"Row type counts    : E={type_counts['E']}, L={type_counts['L']}, G={type_counts['G']}")

    # Sample variables
    print("\n--- Sample Variables ---")
    for i, vname in enumerate(model.var_names[:5]):
        print(f"  {vname}: obj={model.obj[i]:.4f}, bounds=[{model.bounds_lb[i]}, {model.bounds_ub[i]}]")

    # Sample constraints
    print("\n--- Sample Constraints ---")
    for i, rname in enumerate(model.row_names[:5]):
        print(f"  {rname}: type={model.row_types[i]}, RHS={model.rhs[i]:.4f}")

    # Sample coefficients
    print("\n--- Sample Coefficients ---")
    items = list(model.coeffs.items())[:8]
    for (cidx, ridx), val in items:
        print(f"  ({model.var_names[cidx]}, {model.row_names[ridx]}) = {val:.6f}")

    # Bounds summary
    print("\n--- Bounds Summary ---")
    finite_ub = sum(1 for ub in model.bounds_ub if ub != float("inf"))
    non_zero_lb = sum(1 for lb in model.bounds_lb if lb != 0.0)
    print(f"  Variables with finite upper bound : {finite_ub}")
    print(f"  Variables with non-zero lower bound: {non_zero_lb}")
    print(f"  Default [0, +inf] applied to all variables (no BOUNDS section in AFIRO).")

    # Unsupported features check
    print("\n--- Unsupported Features Check ---")
    print("  RANGES section     : not supported (not present in AFIRO)")
    print("  SOS sets           : not supported")
    print("  QUADOBJ section    : not supported")
    print("  Integer variables  : not supported (MARKER/INTORG/INTEND)")
    print("  Multiple RHS/BOUNDS: partially supported (uses first vector only)")


if __name__ == "__main__":
    main()
