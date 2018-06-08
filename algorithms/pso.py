import random

import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


def run_global_best_pso(n_dims, test_func, n_inds, n_gens, lower_bound, upper_bound,
			initial_positions=None, random_seed=12345,
			c1=0.5, c2=0.3, w=0.9
	):
	np.random.seed(random_seed)

	options = {'c1':c1, 'c2':c2, 'w':w}
	bounds = (np.array([lower_bound] * n_dims), np.array([upper_bound] * n_dims))

	optimizer = ps.single.GlobalBestPSO(n_particles=n_inds, dimensions=n_dims, bounds=bounds, options=options)
	if initial_positions is not None:
		optimizer.pos = np.array(initial_positions).copy()

	stats = optimizer.optimize(test_func, iters=n_gens)
	pos_history = optimizer.get_pos_history

	history = list()
	history.append( {'gen': 0, 'individuals': initial_positions} )	# TODO: better to do it inside pyswarms
	for g in range(n_gens):
		solutions = list()
		#fitnesses = list()	# TODO
		for i in range(n_inds):
			solutions.append(pos_history[g][i].tolist())	# convert from np.array to list
		history.append( {'gen': g+1, 'individuals': solutions} )

	return history


if __name__ == "__main__":
	n_inds = 100
	n_gens = 10000

	n_dims = 8
	lower_bound = -5.12
	upper_bound = 5.12
	test_func = fx.rastrigin_func

	initial_positions = [[random.uniform(lower_bound, upper_bound) for _ in range(n_dims)] for _ in range(n_inds)]

	results = run_global_best_pso(n_dims=n_dims, test_func=test_func,
					n_inds=n_inds, n_gens=n_gens,
					lower_bound=lower_bound, upper_bound=upper_bound,
					initial_positions=initial_positions)


