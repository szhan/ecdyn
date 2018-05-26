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

import random
from operator import itemgetter
from collections import OrderedDict

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


def run_simple_genetic_algorithm(n_dims, test_func, test_lb, test_ub,\
				n_inds, n_gens, test_min_goal=0,\
				cx_pb=0.5, mut_pb=0.2, ind_pb=0.05,\
				gauss_mu=0, gauss_sigma=5, tourn_size=3,\
				random_seed=12345
	):
	"""
	A simple genetic algorithm to solve a minimization problem.
	The code is adopted from the GA implementation to solve the OneMax problem in DEAP.
	"""
	
	# set up
	random.seed(random_seed)
	
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMin)
	
	toolbox = base.Toolbox()
	toolbox.register("attr_float_uniform", random.uniform, test_lb, test_ub)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float_uniform, n_dims)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("evaluate", test_func)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutGaussian, mu=gauss_mu, sigma=gauss_sigma, indpb=ind_pb)
	toolbox.register("select", tools.selTournament, tournsize=tourn_size)
	
	# record individuals, sorted by fitness, across generations
	history = list()	# list of dictionaries
	
	# initialize population
	pop = toolbox.population(n=n_inds)
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	fits = [ind.fitness.values[0] for ind in pop]
	
	sorted_fitness = fits.sort()
	indices_sorted_fitness = sorted(range(len(fits)), key=lambda k: fits[k])
	individuals_sorted_fitness = itemgetter(*indices_sorted_fitness)(pop)
	
	history.append({'gen':0, 'individuals':individuals_sorted_fitness, 'fitness':sorted_fitness})	
	
	for g in range(1, n_gens + 1):
		print("Generation %i" % g)
		
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
		sorted_fitness = fits.sort()
		indices_sorted_fitness = sorted(range(len(fits)), key=lambda k: fits[k])
		individuals_sorted_fitness = itemgetter(*indices_sorted_fitness)(pop)
		
		history.append({'gen':g, 'individuals':individuals_sorted_fitness, 'fitness':sorted_fitness})	
		
		if min(fits) <= test_min_goal:
			break
	
	return history


if __name__ == "__main__":
	run_simple_genetic_algorithm(n_dims=3,\
					test_func=benchmarks.ackley, test_lb=-5, test_ub=5,\
					n_inds=5, n_gens=10)


