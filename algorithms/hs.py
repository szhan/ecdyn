#!/usr/bin/env python

"""
    Copyright (c) 2013, Los Alamos National Security, LLC
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
      disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Los Alamos National Security, LLC nor the names of its contributors may be used to endorse or
      promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
    THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import random

from pyharmonysearch import ObjectiveFunctionInterface, harmony_search
from deap import benchmarks


class ObjectiveFunction(ObjectiveFunctionInterface):
	def __init__(self, n_dims, obj_func, lower_bound, upper_bound,
			n_inds, n_gens, random_seed=12345):
		self._n_dims = n_dims
		self._lower_bounds = [float(lower_bound)] * self._n_dims
		self._upper_bounds = [float(upper_bound)] * self._n_dims
		self._variable = [True] * self._n_dims
		self._obj_func = obj_func
		self._random_seed = random_seed

		# define all input parameters
		self._maximize = False  # do we maximize or minimize?
		self._max_imp = n_inds * n_gens  # number of evaluations
		self._hms = n_inds  # harmony memory size
		self._hmcr = 0.75  # harmony memory considering rate
		self._par = 0.5  # pitch adjusting rate
		self._mpap = 0.25  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
		self._mpai = 2  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only

	def get_fitness(self, vector):
		return self._obj_func(vector)[0]

	def get_value(self, i, index=None):
		return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

	def get_lower_bound(self, i):
		return self._lower_bounds[i]

	def get_upper_bound(self, i):
		return self._upper_bounds[i]

	def is_variable(self, i):
		return self._variable[i]

	def is_discrete(self, i):
		# all variables are continuous
		return False

	def get_num_parameters(self):
		return len(self._lower_bounds)

	def use_random_seed(self):
		return hasattr(self, '_random_seed') and self._random_seed

	def get_random_seed(self):
		return self._random_seed

	def get_max_imp(self):
		return self._max_imp

	def get_hmcr(self):
		return self._hmcr

	def get_par(self):
		return self._par

	def get_hms(self):
		return self._hms

	def get_mpai(self):
		return self._mpai

	def get_mpap(self):
		return self._mpap

	def maximize(self):
		return self._maximize


def run_harmony_search(n_dims, test_func, lower_bound, upper_bound, n_inds, n_gens,
			initial_positions=None, random_seed=12345):
	# check input
	assert lower_bound < upper_bound, "Lower bound must be smaller than upper bound."

	if initial_positions is not None:
		assert len(initial_positions) == n_inds
		for position in initial_positions:
			assert len(position) == n_dims
			assert max(position) <= upper_bound
			assert min(position) >= lower_bound

	obj_fun = ObjectiveFunction(n_dims=n_dims, obj_func=test_func,
					lower_bound=lower_bound, upper_bound=upper_bound,
					n_inds=n_inds, n_gens=n_gens,
					random_seed=random_seed)

	result = harmony_search(obj_fun, num_processes=1, num_iterations=1,
				initial_harmonies=initial_positions)

	history = list()
	for g in range(n_gens + 1):
		solutions = list()
		fitnesses = list()
		for i in range(n_inds):
			solutions.append(result.harmony_histories[0][g]['harmonies'][i][0])
			fitnesses.append(result.harmony_histories[0][g]['harmonies'][i][1])
		history.append({'gen': g, 'individuals': solutions, 'fitness': fitnesses})

	return history


if __name__ == '__main__':
	n_inds = 100
	n_gens = 10000

	n_dims = 3
	lower_bound = -5.12
	upper_bound = 5.12
	test_func = benchmarks.rastrigin	# from DEAP, which returns a tuple

        initial_positions = [[random.uniform(lower_bound, upper_bound) for _ in range(n_dims)] for _ in range(n_inds)]
	print 'Initial solution: {}'.format(initial_positions)

	results = run_harmony_search(n_dims=n_dims, test_func=test_func,
					lower_bound=lower_bound, upper_bound=upper_bound,
					n_inds=n_inds, n_gens=n_gens,
					initial_positions=initial_positions)

	assert len(results) == n_gens + 1

	best_solution = None
	best_fitness = None
	for i in range(n_inds):
		if best_solution is None or results[-1]['fitness'][i] < best_fitness:
			best_solution = results[-1]['individuals'][i]
			best_fitness = results[-1]['fitness'][i]

	print 'Best solution: {}\nBest fitness: {}'.format(best_solution, best_fitness)


