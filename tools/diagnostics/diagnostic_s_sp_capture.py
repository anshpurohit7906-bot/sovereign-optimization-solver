from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Global storage for captured S_sp matrices
captured_s_sp = {}

# Original functions
_original_splu = splu
_original_factor_reduced_system = None
_original_splu_regularized = None

# Capture counters
_capture_count = 0
# From the output, the AFIRO solve took 9 iterations total
# We'll capture early iterations 1, 2, 3 and also check later ones if they exist
_capture_iterations = set([1, 2, 3, 5, 8])

# Global splu call counter
splu_call_count = 0

def _capture_splu(*args, **kwargs):
    """Wrapper around splu to capture S_sp before factorization and timing"""
    global _capture_count, captured_s_sp, splu_call_count
    
    # Increment total call counter
    splu_call_count += 1
    
    # Debug: Print every call to see if we're being hit
    print(f"DEBUG: _capture_splu called with {len(args)} args, kwargs count={len(kwargs) if kwargs else 0}")
    
    if len(args) > 0:
        print(f"DEBUG: First arg type: {type(args[0])}")
        # Check for both CSR and CSC formats
        if isinstance(args[0], (sp.csr_matrix, sp.csc_matrix)):
            s_matrix = args[0]
            print(f"DEBUG: Matrix shape: {s_matrix.shape}, nnz: {s_matrix.nnz}, format: {s_matrix.format}")
            
            # Convert to CSR if it's CSC for consistency
            if s_matrix.format == 'csc':
                s_matrix = s_matrix.tocsr()
            
            if _capture_count in _capture_iterations:
                # Start timing factorization
                import time
                start_time = time.perf_counter()
                
                # Call the original splu
                result = _original_splu(*args, **kwargs)
                
                # End timing
                end_time = time.perf_counter()
                factorization_time = end_time - start_time
                
                # Store results including SuperLU object details
                captured_s_sp[_capture_count] = {
                    'shape': s_matrix.shape,
                    'nnz': s_matrix.nnz,
                    'format': s_matrix.format,
                    'indptr': s_matrix.indptr.copy(),
                    'indices': s_matrix.indices.copy(),
                    'data': s_matrix.data.copy(),
                    'factorization_time': factorization_time,
                    'superlu_result': result,
                    'L_nnz': result.L.nnz if hasattr(result, 'L') else None,
                    'U_nnz': result.U.nnz if hasattr(result, 'U') else None,
                    'perm_r': list(result.perm_r) if hasattr(result, 'perm_r') else None,
                    'perm_c': list(result.perm_c) if hasattr(result, 'perm_c') else None
                }
                print(f"CAPTURED iteration {_capture_count}: shape={s_matrix.shape}, nnz={s_matrix.nnz}, factorization_time={factorization_time:.6f}s")
    
    _capture_count += 1
    return _original_splu(*args, **kwargs)

def patch_linear_system():
    """Temporarily patch the linear system module"""
    global _original_splu
    
    # Import the splu function directly from the module where it's used
    from opticore.lp.linear_system import splu as original_splu
    _original_splu = original_splu
    
    # Get the module where splu is imported
    import opticore.lp.linear_system as ls_module
    ls_module.splu = _capture_splu
    
    # Also need to patch the imported reference in scipy.sparse.linalg for imports
    import scipy.sparse.linalg
    if hasattr(scipy.sparse.linalg, 'splu'):
        original_scipy_splu = scipy.sparse.linalg.splu
        scipy.sparse.linalg.splu = _capture_splu
    
    from opticore.lp.mehrotra import solve_lp
    from opticore.numerical_model import load_numeric_mps
    
    return solve_lp, load_numeric_mps, original_scipy_splu

def restore_linear_system(original_scipy_splu=None):
    """Restore original functions"""
    import opticore.lp.linear_system as ls_module
    import scipy.sparse.linalg
    
    ls_module.splu = _original_splu
    
    if original_scipy_splu is not None and hasattr(scipy.sparse.linalg, 'splu'):
        scipy.sparse.linalg.splu = original_scipy_splu

def compare_matrices(matrix1, matrix2, name1, name2):
    """Compare two sparse matrices for structural and numerical differences"""
    results = {
        'shape_match': matrix1.shape == matrix2.shape,
        'nnz_match': matrix1.nnz == matrix2.nnz,
        'structurally_identical': (
            np.array_equal(matrix1.indptr, matrix2.indptr) and
            np.array_equal(matrix1.indices, matrix2.indices)
        ),
        'numerically_identical': np.array_equal(matrix1.data, matrix2.data),
        'max_data_diff': float(np.max(np.abs(matrix1.data - matrix2.data))) if matrix1.nnz == matrix2.nnz else None
    }
    return results

def analyze_superlu_factors():
    """Analyze SuperLU factorization properties"""
    print("\n" + "=" * 50)
    print("SUPERLU FACTORIZATION ANALYSIS")
    print("=" * 50)
    
    if not captured_s_sp:
        print("ERROR: No S_sp matrices were captured!")
        return
    
    sorted_captures = sorted(captured_s_sp.items())
    
    print(f"\nTOTAL SPLU() CALLS: {splu_call_count}")
    
    print(f"\nSUPERLU FACTORIZATION TIMINGS:")
    for iter_num, data in sorted_captures:
        factorization_time = data.get('factorization_time', None)
        if factorization_time is not None:
            print(f"  Iteration {iter_num}: {factorization_time:.6f}s")
    
    print(f"\nL and U NONZERO COUNTS:")
    L_nnz_values = []
    U_nnz_values = []
    
    for iter_num, data in sorted_captures:
        L_nnz = data.get('L_nnz', None)
        U_nnz = data.get('U_nnz', None)
        L_nnz_values.append(L_nnz)
        U_nnz_values.append(U_nnz)
        print(f"  Iteration {iter_num}: L.nnz={L_nnz}, U.nnz={U_nnz}")
    
    # Check if L.nnz values are stable
    if all(x == L_nnz_values[0] for x in L_nnz_values if x is not None):
        print(f"  ✓ L.nnz STABLE across iterations: {L_nnz_values[0]}")
    else:
        print(f"  ✗ L.nnz CHANGED: {L_nnz_values}")
    
    # Check if U.nnz values are stable
    if all(x == U_nnz_values[0] for x in U_nnz_values if x is not None):
        print(f"  ✓ U.nnz STABLE across iterations: {U_nnz_values[0]}")
    else:
        print(f"  ✗ U.nnz CHANGED: {U_nnz_values}")
    
    print(f"\nPERMUTATION VECTORS:")
    perm_r_values = []
    perm_c_values = []
    
    for iter_num, data in sorted_captures:
        perm_r = data.get('perm_r', None)
        perm_c = data.get('perm_c', None)
        perm_r_values.append(perm_r)
        perm_c_values.append(perm_c)
        print(f"  Iteration {iter_num}: perm_r={perm_r}, perm_c={perm_c}")
    
    # Check if permutation vectors are stable
    if all(r == perm_r_values[0] for r in perm_r_values if r is not None):
        print(f"  ✓ perm_r STABLE across iterations")
    else:
        print(f"  ✗ perm_r CHANGED across iterations")
    
    if all(c == perm_c_values[0] for c in perm_c_values if c is not None):
        print(f"  ✓ perm_c STABLE across iterations")
    else:
        print(f"  ✗ perm_c CHANGED across iterations")
    
    print(f"\nFACTORIZATION TIME STABILITY:")
    factorization_times = [data['factorization_time'] for _, data in sorted_captures]
    if len(factorization_times) > 1:
        min_time = min(factorization_times)
        max_time = max(factorization_times)
        relative_variation = (max_time - min_time) / min_time * 100
        
        print(f"  Min time: {min_time:.6f}s")
        print(f"  Max time: {max_time:.6f}s")
        print(f"  Relative variation: {relative_variation:.2f}%")
        
        if relative_variation < 5.0:  # Less than 5% variation
            print(f"  ✓ Factorization times are relatively stable")
        else:
            print(f"  ✗ Factorization times vary significantly")

def solve_with_timed_superlu(fac, b):
    """Solve using SuperLU with timing"""
    import time
    start_time = time.perf_counter()
    
    result = fac.schur_lu.solve(b)
    
    end_time = time.perf_counter()
    solve_time = end_time - start_time
    
    return result, solve_time

def main():
    print("OPTICORE S_sp DIAGNOSTIC SCRIPT")
    print("=" * 50)
    
    # Reset capture count to help debugging
    global _capture_count
    _capture_count = 0
    
    original_scipy_splu = None
    try:
        print("Patching linear system modules...")
        solve_lp, load_numeric_mps, original_scipy_splu = patch_linear_system()
        
        print("Loading AFIRO.mps and solving...")
        mps = ROOT / 'data' / 'afiro.mps'
        lp = load_numeric_mps(mps)
        
        # Solve with verbose to see iteration progress
        result = solve_lp(lp, tol=1e-8, max_iter=200, verbose=True)
        
        print(f"\nSolve completed: {result.status} in {result.iterations} iterations")
        
    except Exception as e:
        print(f"Error during solve: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nRestoring original functions...")
        restore_linear_system(original_scipy_splu)
    
    print("\n" + "=" * 50)
    print("CAPTURE ANALYSIS")
    print("=" * 50)
    
    if not captured_s_sp:
        print("ERROR: No S_sp matrices were captured!")
        return
    
    sorted_captures = sorted(captured_s_sp.items())
    
    print(f"\nCAPTURE METHOD: Runtime patching of scipy.sparse.linalg.splu")
    print(f"ITERATION SAMPLES: {list(captured_s_sp.keys())}")
    
    print(f"\nS_sp SHAPE/NNZ:")
    for iter_num, data in sorted_captures:
        print(f"  Iteration {iter_num}: shape={data['shape']}, nnz={data['nnz']}")
    
    print(f"\nSTRUCTURAL PATTERN RESULT:")
    structure_constant = True
    for i in range(len(sorted_captures) - 1):
        iter1, data1 = sorted_captures[i]
        iter2, data2 = sorted_captures[i + 1]
        
        comp = compare_matrices(
            sp.csr_matrix((data1['data'], data1['indices'], data1['indptr']), shape=data1['shape']),
            sp.csr_matrix((data2['data'], data2['indices'], data2['indptr']), shape=data2['shape']),
            f"iter_{iter1}", f"iter_{iter2}"
        )
        
        print(f"  Iter {iter1} vs {iter2}:")
        print(f"    Shape identical: {comp['shape_match']}")
        print(f"    NNZ identical: {comp['nnz_match']}")
        print(f"    Structurally identical: {comp['structurally_identical']}")
        print(f"    Numerically identical: {comp['numerically_identical']}")
        
        if not comp['structurally_identical']:
            structure_constant = False
            print(f"    MAX DATA DIFFERENCE: {comp['max_data_diff']:.3e}")
    
    print(f"\nNUMERICAL DATA RESULT:")
    all_numerically_same = True
    for i in range(len(sorted_captures)):
        iter1, data1 = sorted_captures[i]
        for j in range(i + 1, len(sorted_captures)):
            iter2, data2 = sorted_captures[j]
            
            comp = compare_matrices(
                sp.csr_matrix((data1['data'], data1['indices'], data1['indptr']), shape=data1['shape']),
                sp.csr_matrix((data2['data'], data2['indices'], data2['indptr']), shape=data2['shape']),
                f"iter_{iter1}", f"iter_{iter2}"
            )
            
            if not comp['numerically_identical']:
                all_numerically_same = False
                print(f"  Iter {iter1} vs {iter2}: Values differ (max diff: {comp['max_data_diff']:.3e})")
    
    print(f"\nFINAL CONCLUSION:")
    if len(sorted_captures) < 2:
        print("  INSUFFICIENT DATA: Need at least 2 captures to compare")
    else:
        structural_test = True
        for i in range(len(sorted_captures) - 1):
            iter1, data1 = sorted_captures[i]
            iter2, data2 = sorted_captures[i + 1]
            
            comp = compare_matrices(
                sp.csr_matrix((data1['data'], data1['indices'], data1['indptr']), shape=data1['shape']),
                sp.csr_matrix((data2['data'], data2['indices'], data2['indptr']), shape=data2['shape']),
                f"iter_{iter1}", f"iter_{iter2}"
            )
            
            if not comp['structurally_identical']:
                structural_test = False
                break
        
        if structural_test:
            print("  ✓ STRUCTURAL PATTERN IDENTICAL ACROSS ALL CAPTURED ITERATIONS")
        else:
            print("  ✗ STRUCTURAL PATTERN CHANGED ACROSS ITERATIONS")
            
        if all_numerically_same:
            print("  ✓ NUMERICAL VALUES IDENTICAL ACROSS ALL CAPTURED ITERATIONS")
        else:
            print("  ✗ NUMERICAL VALUES CHANGED ACROSS ITERATIONS")
            
        if structural_test and not all_numerically_same:
            print("  → CONCLUSION: STRUCTURE CONSTANT WHILE VALUES CHANGE")
        elif structural_test and all_numerically_same:
            print("  → CONCLUSION: BOTH STRUCTURE AND VALUES CONSTANT")
        elif not structural_test:
            print("  → CONCLUSION: STRUCTURE CHANGED")
        else:
            print("  → CONCLUSION: INCOMPLETE ANALYSIS")
    
    # Add SuperLU analysis
    analyze_superlu_factors()

if __name__ == '__main__':
    main()
