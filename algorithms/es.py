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

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


def run_es_mu_plus_lambda(n_dims, test_func, lower_bound, upper_bound, n_inds, n_gens,
				initial_positions=None, random_seed=12345,
				strat_min=0.1, strat_max=0.5,
				alpha=0.1, c=1.0, indpb=0.1,
				lambda_=10, cxpb=0.6, mutpb=0.3
	):
	"""
	This is the :math:`(\mu + \lambda)` evolutionary algorithm.

	:param mu: The number of individuals to select for the next generation.
			Note that this is n_inds.
	:param lambda\_: The number of children to produce at each generation.
	:param cxpb: The probability that an offspring is produced by crossover.
	:param mutpb: The probability that an offspring is produced by mutation.

	The pseudocode goes as follow ::
		evaluate(population)
		for g in range(n_gens):
			offspring = varOr()
			evaluate(offspring)
			population = select()

	This code is modified from DEAP's eaMuPlusLambda() and varOr().
	"""

	# check input
	assert (cxpb + mutpb) <= 1.0, "The sum of the crossover and mutation probabilities must be <= 1.0."

	if initial_positions is not None:
		assert len(initial_positions) == n_inds
		for position in initial_positions:
			assert len(position) == n_dims
			assert max(position) <= upper_bound
			assert min(position) >= lower_bound

	# set up
	random.seed(random_seed)

	def varOr(population, toolbox, lambda_, cxpb, mutpb):
		""" Apply crossover and mutation to population of solutions. """
		offspring = []
		for _ in xrange(lambda_):
			op_choice = random.random()
			if op_choice < cxpb:		# Crossover
				ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
				ind1, ind2 = toolbox.mate(ind1, ind2)
				del ind1.fitness.values
				offspring.append(ind1)
			elif op_choice < cxpb + mutpb:	# Mutation
				ind = toolbox.clone(random.choice(population))
				ind, = toolbox.mutate(ind)
				del ind.fitness.values
				offspring.append(ind)
			else:				# Reproduction
				offspring.append(random.choice(population))
		return offspring

	def checkStrategy(minstrategy):
		""" Check that strategy meets a minimum, otherwise not enough exploration. """
		def decorator(func):
			def wrapper(*args, **kargs):
				children = func(*args, **kargs)
				for child in children:
					for i, s in enumerate(child.strategy):
						if s < minstrategy:
							child.strategy[i] = minstrategy
				return children
			return wrapper
		return decorator


	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMin, strategy=None)

	toolbox = base.Toolbox()
	toolbox.register("evaluate", test_func)
	toolbox.register("mate", tools.cxESBlend, alpha=alpha)
	toolbox.register("mutate", tools.mutESLogNormal, c=c, indpb=indpb)
	toolbox.register("select", tools.selBest)

	toolbox.decorate("mate", checkStrategy(strat_min))
	toolbox.decorate("mutate", checkStrategy(strat_min))

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

	# always randomly generated initial strategies
	for i in range(n_inds):
		pop[i].strategy = [random.uniform(strat_min, strat_max) for _ in range(n_dims)]

	# evaluate individuals with an invalid fitness
	invalid_ind = [ind for ind in pop if not ind.fitness.valid]
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit
	fits = [ind.fitness.values[0] for ind in pop]

	sorted_fitness = sorted(fits)
	indices_sorted_fitness = sorted(range(len(fits)), key=lambda k: fits[k])
	individuals_sorted_fitness = itemgetter(*indices_sorted_fitness)(pop)

	history.append({'gen':0, 'individuals':individuals_sorted_fitness, 'fitness':sorted_fitness})

	for g in range(1, n_gens + 1):
		offspring = varOr(pop, toolbox, lambda_, cxpb, mutpb)

		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		pop[:] = toolbox.select(pop + offspring, n_inds)
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

	initial_positions = [[random.uniform(lower_bound, upper_bound) for _ in range(n_dims)] for _ in range(n_inds)]

	results = run_es_mu_plus_lambda(n_dims=n_dims, test_func=test_func,
					lower_bound=lower_bound, upper_bound=upper_bound,
					n_inds=n_inds, n_gens=n_gens,
					initial_positions=initial_positions)

	best_solution = None
	best_fitness = None
	for i in range(n_inds):
		if best_solution is None or results[-1]['fitness'][i] < best_fitness:
			best_solution = results[-1]['individuals'][i]
			best_fitness = results[-1]['fitness'][i]

	print 'Best solution: {}\nBest fitness: {}'.format(best_solution, best_fitness)


