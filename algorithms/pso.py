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

import operator
import random

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


test_func = benchmarks.griewank
test_lb = -100
test_ub = 100
test_min_goal = 0

n_dims = 2
n_inds = 300
n_gens = 100

p_min = -6
p_max = 6
s_min = -3
s_max = 3
phi_1 = 2.0
phi_2 = 2.0


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,
    smin=None, smax=None, best=None)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


# TODO: generalize to n dimensions
def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=n_dims, pmin=p_min, pmax=p_max, smin=s_min, smax=s_max)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=phi_1, phi2=phi_2)	# TODO: generalize to n dimensions
toolbox.register("evaluate", test_func)

pop = toolbox.population(n=n_inds)

best = None
for g in range(n_gens):
	for part in pop:
		part.fitness.values = toolbox.evaluate(part)
		if not part.best or part.best.fitness < part.fitness:
			part.best = creator.Particle(part)
			part.best.fitness.values = part.fitness.values
		if not best or best.fitness < part.fitness:
			best = creator.Particle(part)
			best.fitness.values = part.fitness.values
	for part in pop:
		toolbox.update(part, best)
	# TODO: print or store g, pop, and fitnesses

