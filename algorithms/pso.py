import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


def run_global_best_pso(n_dims, test_func,\
			n_inds, n_gens,\
			c1=0.5, c2=0.3, w=0.9
	):
	options = {'c1':c1, 'c2':c2, 'w':w}
	optimizer = ps.single.GlobalBestPSO(n_particles=n_inds, dimensions=n_dims, options=options)
	stats = optimizer.optimize(test_func, iters=n_gens)
	pos_history = optimizer.get_pos_history
	return(pos_history)


if __name__ == "__main__":
	run_global_best_pso(n_dims=3, test_func=fx.rastrigin_func, n_inds=5, n_gens=10)


