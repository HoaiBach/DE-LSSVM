"""	
Implement JADE: Adaptive differential evolution with optional external archive	
"""
from Base import *
import Paras
import time
from multiprocessing import Pool

class JADE:
    def __init__(self, problem, popsize, dims, maxiters, minpos, maxpos, c, p):
        self.problem = problem
        self.popsize = popsize
        self.dims = dims
        self.maxiters = maxiters
        self.c = c
        self.num_top = int(p * self.popsize)
        self.minpos = minpos
        self.maxpos = maxpos
        if self.num_top <= 0:
            self.num_top = 1

        self.initialise()

    def initialise(self):
        self.init_pop()

        self.head_print = 'Min reg %f\n' % Paras.min_reg
        self.head_print += 'Max reg %f\n' % Paras.max_reg
        self.head_print += 'Min loss %f\n' % Paras.min_loss
        self.head_print += 'Max loss %f\n' % Paras.max_loss
        self.head_print += '---------------------------------------------\n'

        if Paras.alg_style == 'embed':
            self.head_print += 'Fitness, Reg, Loss, No_features, Features\n'
        elif Paras.alg_style == 'wrapper':
            self.head_print += 'Fitness, Sel_ratio, Error, No_features, Features\n'
        elif Paras.alg_style == 'filter':
            self.head_print += 'Fitness, Sel_ratio, Score, No_features, Features\n'

        # evaluate the initialized population
        temp = self.evaluate_pop()
        self.fitness = temp[:, 1]
        self.c1 = temp[:, 2]  # c1 = reg if embed, sel_ratio = other
        self.c2 = temp[:, 3]  # c2 = loss if embed, error if wrapper, score if filter

        if self.problem.minimized:
            self.best_idx = np.argmin(self.fitness)
        else:
            self.best_idx = np.argmax(self.fitness)
        self.best_fitness = self.fitness[self.best_idx]
        self.best_c1 = self.c1[self.best_idx]
        self.best_c2 = self.c2[self.best_idx]

    def init_pop(self):
        if Paras.alg_style == 'embed':
            pop = []
            if Paras.init_style == 'interval':
                while len(pop) < self.popsize:
                    for rate in range(1, 101, 1):
                        if len(pop) < self.popsize:
                            pop.append(
                                self.problem.generate_candidate_with_nf(int(rate / 100.0 * self.problem.no_features)))
                        else:
                            break
            else:
                pop = self.minpos + np.random.rand(self.popsize, self.dims) * (self.maxpos - self.minpos)

            self.population = np.array(pop)

            # base on the init population, initialize min and max range for weight part
            max_pos = np.max(self.population[:, :self.problem.no_features + 1])
            min_pos = np.min(self.population[:, self.problem.no_features + 1])
            max_abs_pos = max(abs(max_pos), abs(min_pos))
            self.maxpos[:self.problem.no_features + 1] = [max_abs_pos, ] * (self.problem.no_features + 1)
            self.minpos[:self.problem.no_features + 1] = [-max_abs_pos, ] * (self.problem.no_features + 1)

            # set the upper bound and lower bound for each component in the fitness
            if Paras.fit_normalized:
                import sys
                min_reg = sys.float_info.max
                max_reg = -2
                min_loss = sys.float_info.max
                max_loss = -2
                for ind in self.population:
                    _, reg_ind, loss_ind = self.problem.fitness(ind)
                    min_reg = min(min_reg, reg_ind)
                    max_reg = max(max_reg, reg_ind)
                    min_loss = min(min_loss, loss_ind)
                    max_loss = max(max_loss, loss_ind)
                Paras.min_reg = min_reg
                Paras.max_reg = max_reg
                Paras.min_loss = min_loss
                Paras.max_loss = max_loss
            else:
                Paras.min_reg = 0
                Paras.max_reg = 1
                Paras.min_loss = 0
                Paras.max_loss = 1

        else:
            pop = self.minpos + np.random.rand(self.popsize, self.dims) * (self.maxpos - self.minpos)
            self.population = np.array(pop)

    def core_evaluate(self, index_individual):
        trial_fit, trial_c1, trial_c2 = self.problem.fitness(index_individual[1])
        return [index_individual[0], trial_fit, trial_c1, trial_c2]

    def evaluate_pop(self):
        '''
        Evaluate the current population
        :return: a list where each element is a list [idx, ind_fit, ind_c1, ind_c2]
        '''
        if Paras.parallel:
            indices_pop = [[ind_idx, ind] for ind_idx, ind in enumerate(self.population)]
            pool = Pool(processes=8)
            fitnesses = pool.map(self.core_evaluate, indices_pop)
            pool.close()
        else:
            fitnesses = []
            for idx, ind in enumerate(self.population):
                ind_fit, ind_c1, ind_c2 = self.problem.fitness(ind)
                fitnesses.append([idx, ind_fit, ind_c1, ind_c2])
        return np.array(fitnesses)

    def evolve(self):
        mean_cr = 0.5
        mean_f = 0.5
        std = 0.1
        archive = []
        to_print = self.head_print

        for iteration in range(self.maxiters):
            start = time.time()
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

            if Paras.parallel:
                trial_list = []
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
                        idx_r2 = np.random.randint(self.popsize + len(archive))
                        if idx_r2 != idx and idx_r2 != idx_r1:
                            break
                    if idx_r2 < self.popsize:
                        x_r2 = self.population[idx_r2]
                    else:
                        x_r2 = np.array(archive[idx_r2 - self.popsize])
                    mutant = jade_mutant(x_i=ind, x_b=x_top, x_r1=x_r1, x_r2=x_r2, F=f)
                    trial = jade_crossover(x_i=ind, v=mutant, CR=cr,
                                           minpos=self.minpos, maxpos=self.maxpos)
                    trial_list.append([idx, trial])

                # evaluate all trials parallely
                pool = Pool(processes=8)
                test = pool.map(self.core_evaluate, trial_list)
                pool.close()

                success_cr = []
                success_f = []

                for ele in test:
                    idx, trial_fit, trial_c1, trial_c2 = ele[0], ele[1], ele[2], ele[3]
                    if not self.problem.is_better(self.fitness[idx], trial_fit):
                        archive.append(np.copy(self.population[idx]))
                        self.population[idx] = trial_list[idx][1]
                        self.fitness[idx] = trial_fit
                        self.c1[idx] = trial_c1
                        self.c2[idx] = trial_c2
                        success_cr.append(crs[idx])
                        success_f.append(fs[idx])

                        # check to update the best solution
                        if self.problem.is_better(self.fitness[idx], self.best_fitness):
                            self.best_fitness = self.fitness[idx]
                            self.best_idx = idx
                            self.best_c1 = self.c1[idx]
                            self.best_c2 = self.c2[idx]

            else:
                for idx, ind in enumerate(self.population):
                    # generate new individual
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
                        idx_r2 = np.random.randint(self.popsize + len(archive))
                        if idx_r2 != idx and idx_r2 != idx_r1:
                            break
                    if idx_r2 < self.popsize:
                        x_r2 = self.population[idx_r2]
                    else:
                        x_r2 = np.array(archive[idx_r2 - self.popsize])
                    mutant = jade_mutant(x_i=ind, x_b=x_top, x_r1=x_r1, x_r2=x_r2, F=f)
                    trial = jade_crossover(x_i=ind, v=mutant, CR=cr,
                                           minpos=self.minpos, maxpos=self.maxpos)

                    trial_fit, trial_c1, trial_c2 = self.problem.fitness(trial)

                    if not self.problem.is_better(self.fitness[idx], trial_fit):
                        archive.append(np.copy(ind))
                        self.population[idx] = trial
                        self.fitness[idx] = trial_fit
                        self.c1[idx] = trial_c1
                        self.c2[idx] = trial_c2
                        success_cr.append(cr)
                        success_f.append(f)
                        # check to update the best solution
                        if self.problem.is_better(self.fitness[idx], self.best_fitness):
                            self.best_fitness = self.fitness[idx]
                            self.best_idx = idx
                            self.best_c1 = self.c1[idx]
                            self.best_c2 = self.c2[idx]

            # Maintain the size of archive
            while len(archive) > self.popsize:
                idx_remove = np.random.randint(len(archive))
                archive = np.delete(archive, idx_remove, 0).tolist()
            if len(success_cr) <= 0:
                mean_cr = (1 - self.c) * mean_cr
            else:
                mean_cr = (1 - self.c) * mean_cr + self.c * np.mean(success_cr)
            if np.sum(success_f) == 0:
                mean_f = (1 - self.c) * mean_f
            else:
                mean_f = (1 - self.c) * mean_f + self.c * np.sum(np.asarray(success_f) ** 2) / np.sum(
                    success_f)

            # Extract info to print out:
            if Paras.alg_style == 'embed':
                feature_mask = np.copy(self.population[self.best_idx][self.problem.no_features + 1:])
            else:
                feature_mask = np.copy(self.population[self.best_idx])
            assert len(feature_mask) == self.problem.no_features
            feature_sel = np.where(feature_mask > Paras.threshold)[0]
            str_sel = '['
            for feature_idx in feature_sel:
                str_sel += str(feature_idx) + ', '
            str_sel += ']'
            to_print += 'Iteration %d: %f, %f, %f, %d, %s\n' % \
                        (iteration, self.best_fitness, self.best_c1, self.best_c2, len(feature_sel), str_sel)
            iter_time = time.time() - start
            print('An iteration %d takes: %f \n' % (iteration, iter_time))

        return self.population[self.best_idx], self.best_fitness, to_print

