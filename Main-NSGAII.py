import JADE_Embed
import Problem
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
import Paras
import time
import array
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
import random


def uniform(low, up, size=None):
    try:
        return [np.random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [np.random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def evaluate(sol, problem):
    fitness, reg_weight, loss = problem.fitness(sol)
    return reg_weight, loss


def get_knee_point(front):
    uniq_front = []
    for sol in front:
        add = True
        for uniq_sol in uniq_front:
            if uniq_sol.fitness.values == sol.fitness.values:
                add = False
                break
        if add:
            uniq_front.append(sol)

    no_obj = len(uniq_front[0].fitness.values)
    import sys
    min_max = [[sys.float_info.max,]*no_obj, [-sys.float_info.max,]*no_obj]
    end_points = [sys.float_info.max,]
    pop_fit = []
    for sol in uniq_front:
        fitnesses = sol.fitness.values
        pop_fit.append(np.array(fitnesses))
        min_max[0] = np.minimum(min_max[0], fitnesses)
        min_max[1] = np.maximum(min_max[1], fitnesses)
    min_indices = [-1,]*no_obj
    for obj_idx, min_obj in enumerate(min_max[0]):
        for sol_idx, sol in enumerate(uniq_front):
            if sol.fitness.values[obj_idx] == min_obj:
                min_indices[obj_idx] = sol_idx
                break
    pop_fit = (pop_fit-min_max[0])/(min_max[1]-min_max[0])

    dis = [abs(np.sum(fit)-1) for fit in pop_fit]
    max_idx = np.argmax(dis)
    return uniq_front[max_idx]

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    max_iterations = 100
    CXPB = 0.9
    pop_size = 100
    to_print = 'Maximum number of iterations: %d \n' % max_iterations
    to_print += 'Population size: %d \n' % pop_size
    run = int(sys.argv[2])
    Paras.fit_normalized = True

    to_print += 'Threshold: %f \n' % Paras.threshold
    to_print += '============================================\n'

    seed = 1617*run
    np.random.seed(seed)
    random.seed(seed)

    #load data
    mat = scipy.io.loadmat('/home/nguyenhoai2/Grid/data/FSMathlab/'+dataset+'.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]

    # testing
    # X = -1 + np.random.rand(10, 5)*2
    # W = np.array([-2, 0.01, 0.2, 0.01, 0])
    # y = np.ravel(np.dot(X, np.reshape(W, (5, 1))))
    # y[y > 0] = 1
    # y[y < 0] = 0

    # ensure that y label start from 0, not 1
    num_class, count = np.unique(y, return_counts=True)
    n_classes = np.unique(y).shape[0]
    assert(n_classes == 2)
    min_class = np.min(count)
    unique_classes = np.unique(y)
    y[y == unique_classes[0]] = -1
    y[y == unique_classes[1]] = 1
    y = np.int8(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1617, stratify=y)

    # normalize so all values in the range -1 and 1
    X_min = np.min(X_train, axis=0)
    X_max = np.max(X_train, axis=0)
    non_dup = np.where(X_min != X_max)[0]
    X_min = X_min[non_dup]
    X_max = X_max[non_dup]
    X_train = X_train[:, non_dup]
    X_test = X_test[:, non_dup]
    X_train = 2*(X_train - X_min) / (X_max - X_min)-1
    X_test = 2*(X_test - X_min) / (X_max - X_min)-1
    no_features = X_train.shape[1]

    start = time.time()
    prob = Problem.FS_LSSVM(X_train, y_train)

    pop_temp = []
    while len(pop_temp) < pop_size:
        for rate in range(1, 101, 1):
            if len(pop_temp) < pop_size:
                pop_temp.append(prob.generate_candidate_with_nf(int(rate / 100.0 * no_features)))
            else:
                break
    pop_temp = np.array(pop_temp)

    # base on the init population, initialize min and max range for weight part
    max_pos = np.max(pop_temp[:, :no_features + 1])
    min_pos = np.min(pop_temp[:, :no_features + 1])
    max_abs_pos = max(abs(max_pos), abs(min_pos))
    min_pos = [-1, ]*(no_features+1)+[0, ]*no_features
    max_pos = [1, ]*(no_features+1)+[1, ]*no_features
    max_pos[:no_features + 1] = [max_abs_pos, ] * (no_features + 1)
    min_pos[:no_features + 1] = [-max_abs_pos, ] * (no_features + 1)

    # set the upper bound and lower bound for each component in the fitness
    if Paras.fit_normalized:
        import sys

        min_reg = sys.float_info.max
        max_reg = -2
        min_loss = sys.float_info.max
        max_loss = -2
        for ind in pop_temp:
            _, reg_ind, loss_ind = prob.fitness(ind)
            min_reg = min(min_reg, reg_ind)
            max_reg = max(max_reg, reg_ind)
            min_loss = min(min_loss, loss_ind)
            max_loss = max(max_loss, loss_ind)
        Paras.min_reg = min_reg
        Paras.max_reg = max_reg
        Paras.min_loss = min_loss
        Paras.max_loss = max_loss
    else:
        Paras.min_reg = 0.0
        Paras.max_reg = 1.0
        Paras.min_loss = 0.0
        Paras.max_loss = 1.0

    dim = 2*no_features+1

    # initialize deap
    creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
    creator.create('Individual', array.array, typecode='d', fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('att_float', uniform, min_pos, max_pos, dim)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.att_float)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, problem=prob)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=min_pos, up=max_pos, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=min_pos, up=max_pos, eta=20.0, indpb=1.0 / dim)
    toolbox.register("select", tools.selNSGA2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=pop_size)
    assert len(pop) == len(pop_temp)
    for ind, ind_tmp in zip(pop, pop_temp):
        for idx in range(len(ind)):
            ind[idx] = ind_tmp[idx]

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, max_iterations):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, pop_size)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        # print(logbook.stream)

    # print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))
    nd_front = tools.sortNondominated(pop, len(pop))[0]
    selected_features = []

    # for sol in nd_front:
    #     weight = sol[0:no_features]
    #     b = sol[no_features:no_features + 1]
    #     mask = np.array(sol[no_features + 1:])
    #     f_selected = np.where(mask > Paras.threshold)[0]
    #     for f in f_selected:
    #         if not f in selected_features:
    #             selected_features.append(f)
    #     if len(f_selected) > 0:
    #         X_train_sel = X_train[:, f_selected]
    #         X_test_sel = X_test[:, f_selected]
    #         clf = svm.LinearSVC(random_state=1617, C=1.0)
    #         clf.fit(X_train_sel, y_train)
    #         sel_acc = accuracy_score(y_test, clf.predict(X_test_sel))
    #         train_sel_acc = accuracy_score(y_train, clf.predict(X_train_sel))
    #         print('Reg_weight = %.4f, loss = %.4f, NF = %f, Training Acc = %.4f, Testing Acc = %.4f\n'
    #               % (sol.fitness.values[0],
    #                  sol.fitness.values[1],
    #                  len(f_selected),
    #                 train_sel_acc, sel_acc))

    knee_sol = get_knee_point(nd_front)
    weight = knee_sol[0:no_features]
    b = knee_sol[no_features:no_features + 1]
    mask = np.array(knee_sol[no_features + 1:])
    f_selected = np.where(mask > Paras.threshold)[0]
    if len(f_selected) > 0:
        X_train_sel = X_train[:, f_selected]
        X_test_sel = X_test[:, f_selected]
        clf = svm.LinearSVC(random_state=1617, C=1.0)
        clf.fit(X_train_sel, y_train)
        sel_acc = accuracy_score(y_test, clf.predict(X_test_sel))
        train_sel_acc = accuracy_score(y_train, clf.predict(X_train_sel))
        print('Knee point: Reg_weight = %.4f, loss = %.4f, NF = %f, Training Acc = %.4f, Testing Acc = %.4f\n'
              % (knee_sol.fitness.values[0],
                 knee_sol.fitness.values[1],
                 len(f_selected),
                train_sel_acc, sel_acc))

    # selected_features = np.array(selected_features)
    # X_train_sel = X_train[:, selected_features]
    # X_test_sel = X_test[:, selected_features]
    # clf = svm.LinearSVC(random_state=1617, C=1.0)
    # clf.fit(X_train_sel, y_train)
    # sel_acc = accuracy_score(y_test, clf.predict(X_test_sel))
    # train_sel_acc = accuracy_score(y_train, clf.predict(X_train_sel))
    # print('Combination: NF = %f, Training Acc = %.4f, Testing Acc = %.4f\n'
    #       % (len(selected_features),
    #          train_sel_acc, sel_acc))


