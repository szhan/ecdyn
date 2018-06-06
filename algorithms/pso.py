import itertools

import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


def run_global_best_pso(n_dims, test_func, n_inds, n_gens,\
			initial_positions=None,\
			c1=0.5, c2=0.3, w=0.9,\
			random_seed=12345
	):
	options = {'c1':c1, 'c2':c2, 'w':w}

	optimizer = ps.single.GlobalBestPSO(n_particles=n_inds, dimensions=n_dims, options=options)
	if initial_positions is not None:
		optimizer.pos = np.array(initial_positions).copy()

	stats = optimizer.optimize(test_func, iters=n_gens)
	pos_history = optimizer.get_pos_history

	return(pos_history)


if __name__ == "__main__":
	n_inds = 100
	n_gens = 10000

	n_dims = 3
	lower_bound = -5.12
	upper_bound = 5.12
	test_func = fx.rastrigin_func

	initial_positions = list(itertools.repeat([2.0, 3.0, -0.02], n_inds))

	results = run_global_best_pso(n_dims=n_dims, test_func=test_func,\
					n_inds=n_inds, n_gens=n_gens,\
					initial_positions=initial_positions)


