import numpy as np
import math
import base
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from numpy.linalg import norm
import Paras


class Problem:
    def __init__(self, minimized):
        self.minimized = minimized

    def fitness(self, sol):
        return 10*sol.shape[0]+np.sum(sol**2-10*np.cos(2*math.pi*sol))
        # return np.sum(sol**2)

    def worst_fitness(self):
        w_f = float('inf') if self.minimized else float('-inf')
        return w_f

    def is_better(self, first, second):
        if self.minimized:
            return first < second
        else:
            return first > second


class F1(Problem):

    def __init__(self):
        Problem.__init__(self, minimized=True)

    def fitness(self, sol):
        return np.sum(sol**2)


class F2(Problem):

    def __init__(self):
        Problem.__init__(self, minimized=True)

    def fitness(self, sol):
        abs_sol = np.abs(sol)
        return np.sum(abs_sol) + np.prod(abs_sol)


class F3(Problem):

    def __init__(self):
        Problem.__init__(self, minimized=True)

    def fitness(self, sol):
        square = sol**2
        sum_up_to = [np.sum(square[:i+1]) for i in range(len(square))]
        return np.sum(sum_up_to)


class FS_LSSVM(Problem):

    def __init__(self, X_train, y_train):
        Problem.__init__(self, minimized=True)
        self.X_train = X_train
        self.y_train = np.copy(y_train)
        self.no_instances, self.no_features = X_train.shape[0], X_train.shape[1]

    def fitness(self, sol):
        weight = sol[0:self.no_features]
        b = sol[self.no_features:self.no_features+1]
        mask = sol[self.no_features+1:]
        mask = (mask-Paras.minpos)/(Paras.maxpos-Paras.minpos)
        selected_index = np.where(mask > Paras.theta)[0]
        unselected_index = np.where(mask <= Paras.theta)[0]

        X_train_sel = self.X_train[:, selected_index]
        weight_sel = weight[selected_index]
        # sol[unselected_index] = 0

        no_selected = len(selected_index)
        reg_weight = 0.5*norm(weight_sel)
        output = np.dot(X_train_sel, np.reshape(weight_sel, (no_selected, 1)))+b
        output = np.ravel(output)
        loss = 0.5*Paras.alpha*norm(output*self.y_train-1)
        pen_sel = Paras.beta*no_selected
        if no_selected == 0:
            fitness = self.worst_fitness()
        else:
            fitness = reg_weight + loss + pen_sel

        return fitness

    # def gradient_step(self, sol):
    #     weight = sol[0:self.no_features]
    #     b = sol[self.no_features:self.no_features+1]
    #
    #     output = np.dot(self.X_train, np.reshape(weight, (self.no_features, 1))) + b
    #     output = np.ravel(output)
    #     step = Paras.l_rate* Paras.alpha*(output*self.y_train-1)
    #     weight = weight - step

