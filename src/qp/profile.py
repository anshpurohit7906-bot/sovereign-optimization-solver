"""cProfile entry point for the QP solver."""
import cProfile, pstats
from .benchmark import generate_sparse_qp
from .solver import solve_qp

def profile_qp(n=1000, output='qp_profile.prof'):
    problem=generate_sparse_qp(n,seed=26119+n)
    prof=cProfile.Profile(); prof.enable(); result=solve_qp(problem,max_iterations=50,tol=1e-6,sparse=True); prof.disable()
    prof.dump_stats(output); pstats.Stats(prof).sort_stats('cumtime').print_stats(30)
    return result

if __name__=='__main__': profile_qp()
