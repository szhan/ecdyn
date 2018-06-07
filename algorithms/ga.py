#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import itertools
import random
from operator import itemgetter
from collections import OrderedDict

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


def run_simple_genetic_algorithm(n_dims, test_func, lower_bound, upper_bound, n_inds, n_gens,\
				initial_positions=None,\
				cx_pb=0.5, mut_pb=0.1, ind_pb=0.05,\
				gauss_mu=0, gauss_sigma=5, tourn_size=5,\
				random_seed=12345
	):
	"""
	A simple genetic algorithm to solve a minimization problem.
	The code is adopted from the GA implementation to solve the OneMax problem in DEAP.
	"""
	
	if initial_positions is not None:
		assert len(initial_positions) == n_inds
		for position in initial_positions:
			assert len(position) == n_dims
			assert max(position) <= upper_bound
			assert min(position) >= lower_bound
	
	# set up
	random.seed(random_seed)
	
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMin)
	
	toolbox = base.Toolbox()
	toolbox.register("evaluate", test_func)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutGaussian, mu=gauss_mu, sigma=gauss_sigma, indpb=ind_pb)
	toolbox.register("select", tools.selTournament, tournsize=tourn_size)
	
	# record individuals, sorted by fitness, across generations
	history = list()	# list of dictionaries
	
	# initialize population
	pop = list()
	if initial_positions is not None:
		for i in range(n_inds):
			pop.append(creator.Individual(initial_positions[i]))
	else:
		for i in range(n_inds):
			random_position = [random.uniform(lower_bound, upper_bound) for _ in range(n_inds)]
			pop.append(creator.Individual(random_position))
	
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	fits = [ind.fitness.values[0] for ind in pop]
	
	sorted_fitness = sorted(fits)
	indices_sorted_fitness = sorted(range(len(fits)), key=lambda k: fits[k])
	individuals_sorted_fitness = itemgetter(*indices_sorted_fitness)(pop)
	
	history.append({'gen':0, 'individuals':individuals_sorted_fitness, 'fitness':sorted_fitness})	
	
	for g in range(1, n_gens + 1):
		offspring = toolbox.select(pop, len(pop))
		offspring = list(map(toolbox.clone, offspring))
		
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < cx_pb:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values
		
		for mutant in offspring:
			if random.random() < mut_pb:
				toolbox.mutate(mutant)
				del mutant.fitness.values
		
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		
		pop[:] = offspring
		fits = [ind.fitness.values[0] for ind in pop]
		
		# sort individuals by ascending fitness
		sorted_fitness = sorted(fits)
		indices_sorted_fitness = sorted(range(len(fits)), key=lambda k: fits[k])
		individuals_sorted_fitness = itemgetter(*indices_sorted_fitness)(pop)
		
		history.append({'gen':g, 'individuals':individuals_sorted_fitness, 'fitness':sorted_fitness})	

	return history


if __name__ == "__main__":
	n_inds = 100
	n_gens = 10000

	n_dims = 3
	lower_bound = -5.12
	upper_bound = 5.12
	test_func = benchmarks.rastrigin

	initial_positions = list(itertools.repeat([2.0, -4.0, 5.0], n_inds))

	results = run_simple_genetic_algorithm(n_dims=n_dims, test_func=test_func,\
						lower_bound=lower_bound, upper_bound=upper_bound,\
						n_inds=n_inds, n_gens=n_gens,\
						initial_positions=initial_positions)

	best_solution = None
	best_fitness = None
	for i in range(n_inds):
		if best_solution is None or\
			results[-1]['fitness'][i] < best_fitness:
			best_solution = results[-1]['individuals'][i]
			best_fitness = results[-1]['fitness'][i]

	print 'Best solution: {}\nBest fitness: {}'.format(best_solution, best_fitness)


