"""
Implement JADE: Adaptive differential evolution with optional external archive
"""
from base import *
import Paras


class JADE:
    def __init__(self, problem, popsize, dims, maxiters, c, p):
        self.problem = problem
        self.popsize = popsize
        self.dims = dims
        self.maxiters = maxiters
        self.c = c
        self.num_top = int(p*self.popsize)
        if self.num_top <= 0:
            self.num_top = 1

        # init population
        self.population = self.initialize_pop()
        self.fitness = np.asarray(self.evaluate())
        if self.problem.minimized:
            self.best_idx = np.argmin(self.fitness)
        else:
            self.best_idx = np.argmax(self.fitness)
        self.best_fitness = self.fitness[self.best_idx]

    def initialize_pop(self):
        return random_init(self.popsize, self.dims, Paras.minpos, Paras.maxpos)

    def evaluate(self):
        return [self.problem.fitness(ind) for ind in self.population]

    def evolve(self):
        mean_cr = 0.5
        mean_f = 0.5
        std = 0.1
        archive = []

        to_print = ''
        for iter in range(self.maxiters):
            success_cr = []
            success_f = []
            crs = np.random.normal(mean_cr, std, self.popsize)
            crs[crs > 1.0] = 1.0
            crs[crs < 0.0] = 0.0
            fs = np.random.normal(mean_f, std, self.popsize)
            fs[fs > 1.0] = 1.0
            fs[fs < 0.0] = 0.0
            if self.problem.minimized:
                top_indices = np.argsort(self.fitness)[:self.num_top]
            else:
                top_indices = np.argsort(self.fitness)[-self.num_top:]

            for idx, ind in enumerate(self.population):
                cr = crs[idx]
                f = fs[idx]
                idx_top = np.random.choice(top_indices)
                x_top = self.population[idx_top]

                while True:
                    idx_r1 = np.random.randint(self.popsize)
                    if idx_r1 != idx:
                        break
                x_r1 = self.population[idx_r1]

                while True:
                    idx_r2 = np.random.randint(self.popsize+len(archive))
                    if idx_r2 != idx and idx_r2 != idx_r1:
                        break
                if idx_r2 < self.popsize:
                    x_r2 = self.population[idx_r2]
                else:
                    x_r2 = np.array(archive[idx_r2-self.popsize])

                mutant = jade_mutant(x_i=ind, x_b=x_top, x_r1=x_r1, x_r2=x_r2, F=f)
                trial = jade_crossover(x_i=ind, v=mutant, CR=cr, minpos=Paras.minpos, maxpos=Paras.maxpos)
                # self.problem.gradient_step(trial)
                trial_fit = self.problem.fitness(trial)

                if not self.problem.is_better(self.fitness[idx], trial_fit):
                    archive.append(np.copy(ind))
                    self.population[idx] = trial
                    self.fitness[idx] = trial_fit
                    success_cr.append(cr)
                    success_f.append(f)
                    # check to update the best solution
                    if self.problem.is_better(self.fitness[idx], self.best_fitness):
                        self.best_fitness = self.fitness[idx]
                        self.best_idx = idx

            # Maintain the size of archive
            while len(archive) > self.popsize:
                idx_remove = np.random.randint(len(archive))
                archive = np.delete(archive, idx_remove, 0).tolist()

            if len(success_cr) <= 0:
                mean_cr = (1 - self.c) * mean_cr
            else:
                mean_cr = (1-self.c)*mean_cr + self.c*np.mean(success_cr)

            if np.sum(success_f) == 0:
                mean_f = (1 - self.c) * mean_f
            else:
                mean_f = (1-self.c)*mean_f + self.c*np.sum(np.asarray(success_f)**2)/np.sum(success_f)

            to_print += 'Iteration %d: %f\n' % (iter, self.best_fitness)
        return self.population[self.best_idx], self.best_fitness, to_print
