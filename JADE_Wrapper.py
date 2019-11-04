"""
Implement JADE: Adaptive differential evolution with optional external archive
"""
from base import *
import Paras


class JADE:
    def __init__(self, problem, popsize, dims, maxiters, minpos, maxpos, c, p):
        self.problem = problem
        self.popsize = popsize
        self.dims = dims
        self.maxiters = maxiters
        self.c = c
        self.num_top = int(p*self.popsize)
        self.minpos = minpos
        self.maxpos = maxpos
        if self.num_top <= 0:
            self.num_top = 1

        # init population
        self.population = self.minpos + np.random.rand(popsize, dims) * (self.maxpos-self.minpos)

        self.head_print = 'Min reg %f\n' % Paras.min_reg
        self.head_print += 'Max reg %f\n' % Paras.max_reg
        self.head_print += 'Min loss %f\n' % Paras.min_loss
        self.head_print += 'Max loss %f\n' % Paras.max_loss
        self.head_print += '---------------------------------------------\n'
        self.head_print += 'Fitness, Sel_ratio, Error, No_features, Features\n'

        # evaluate the initialized population
        self.fitness = []
        self.sel_ratio = []
        self.error = []
        for ind in self.population:
            fit_ind, sel_ratio_ind, err_ind = self.problem.fitness(ind)
            self.fitness.append(fit_ind)
            self.sel_ratio.append(sel_ratio_ind)
            self.error.append(err_ind)
        self.fitness = np.array(self.fitness)
        self.sel_ratio = np.array(self.sel_ratio)
        self.error = np.array(self.error)

        if self.problem.minimized:
            self.best_idx = np.argmin(self.fitness)
        else:
            self.best_idx = np.argmax(self.fitness)
        self.best_fitness = self.fitness[self.best_idx]
        self.best_sel_ratio = self.sel_ratio[self.best_idx]
        self.best_error = self.error[self.best_idx]

    def evolve(self):
        mean_cr = 0.5
        mean_f = 0.5
        std = 0.1
        archive = []

        to_print = self.head_print
        for iter in range(self.maxiters):
            success_cr = []
            success_f = []
            crs = np.random.normal(mean_cr, std, self.popsize)
            crs[crs > 1.0] = 1.0
            crs[crs < 0.0] = 0.0
            fs = np.random.normal(mean_f, std, self.popsize)
            fs[fs > 1.0] = 1.0
            fs[fs < 0.0] = 0.
            sort_idx = np.argsort(self.fitness)

            if self.problem.minimized:
                top_indices = sort_idx[:self.num_top]
            else:
                top_indices = sort_idx[-self.num_top:]

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
                trial = jade_crossover(x_i=ind, v=mutant, CR=cr,
                                       minpos=self.minpos, maxpos=self.maxpos)

                trial_fit, trial_reg, trial_loss = self.problem.fitness(trial)

                if not self.problem.is_better(self.fitness[idx], trial_fit):
                    archive.append(np.copy(ind))
                    self.population[idx] = trial
                    self.fitness[idx] = trial_fit
                    self.sel_ratio[idx] = trial_reg
                    self.error[idx] = trial_loss
                    success_cr.append(cr)
                    success_f.append(f)
                    # check to update the best solution
                    if self.problem.is_better(self.fitness[idx], self.best_fitness):
                        self.best_fitness = self.fitness[idx]
                        self.best_idx = idx
                        self.best_sel_ratio = self.sel_ratio[idx]
                        self.best_error = self.error[idx]


            # now perform local search on the best fitness
            # new_best = self.problem.generate_candidate(self.population[np.random.randint(0, self.popsize)])
            # new_fit = self.problem.fitness(new_best)[0]
            # if self.problem.is_better(new_fit, self.best_fitness):
            #     # if the new solution is better than the best
            #     # replace the best
            #     self.population[self.best_idx] = new_best
            #     self.fitness[self.best_idx] = new_fit
            #     self.best_fitness = new_fit
            # else:
            #     # otherwise, randomly replace a candidate solution
            #     replace_idx = np.random.randint(0, self.popsize)
            #     self.population[replace_idx] = new_best
            #     self.fitness[replace_idx] = new_fit

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

            feature_mask = self.population[self.best_idx]
            assert len(feature_mask) == self.problem.no_features
            feature_sel = np.where(feature_mask > Paras.threshold)[0]
            str_sel = '['
            for feature_idx in feature_sel:
                str_sel += str(feature_idx)+', '
            str_sel += ']'

            to_print += 'Iteration %d: %f, %f, %f, %d, %s\n' % \
                        (iter, self.best_fitness, self.best_sel_ratio, self.best_error, len(feature_sel), str_sel)
        return self.population[self.best_idx], self.best_fitness, to_print
