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

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


"""
A simple genetic algorithm to solve the Griewank problem. The code is adopted from the GA implementation to solve the OneMax problem in DEAP.
"""

random.seed(64)

test_func = benchmarks.griewank
test_lb = -5
test_ub = 5
test_min_goal = 0

n_dims = 2
n_inds = 300
n_gens = 100

cx_pb = 0.5
mut_pb = 0.2
ind_pb = 0.05
gauss_mu = 0
gauss_sigma = 5
tourn_size = 3

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

pop = toolbox.population(n=n_inds)

# initialize fitness values
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
	ind.fitness.values = fit
fits = [ind.fitness.values[0] for ind in pop]

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
	
	# TODO: print or store g, pop, and fitnesses
	
	if min(fits) <= test_min_goal:
		break


