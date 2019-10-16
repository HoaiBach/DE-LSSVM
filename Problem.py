import numpy as np
import math
import base
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
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


class FS_Wrapper(Problem):

    def __init__(self, X, y, clf):
        Problem.__init__(self, minimized=True)
        self.X = X
        self.y = y
        num_class, count = np.unique(self.y, return_counts=True)
        min_class = np.min(count)
        n_splits = min(min_class, 5)
        self.skf = StratifiedKFold(n_splits=n_splits, random_state=1617)
        self.no_features = self.X.shape[1]
        self.clf = clf

    def fitness(self, sol):
        sel_idx = np.where(sol > Paras.theta)[0]
        if len(sel_idx) == 0:
            return self.worst_fitness()
        sum_err = 0
        fold_count = 0
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            self.clf.fit(X_train, y_train)
            err = 1 - accuracy_score(y_test, self.clf.predict(X_test))
            sum_err += err
            fold_count = fold_count+1
        ave_err = sum_err/fold_count
        sel_ratio = float(len(sel_idx))/len(sol)
        fitness = Paras.w_wrapper*ave_err + (1.0-Paras.w_wrapper)*sel_ratio
        return fitness


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
        selected_index = np.where(mask > Paras.theta)[0]
        unselected_index = np.where(mask <= Paras.theta)[0]

        X_train_sel = self.X_train[:, selected_index]
        weight_sel = weight[selected_index]
        # sol[unselected_index] = 0

        no_selected = len(selected_index)
        rate_selected = len(selected_index)#/float(self.no_features)
        pen_sel = Paras.beta*rate_selected
        reg_weight = 0.5*norm(weight_sel)#/(self.no_features**2)
        output = np.dot(X_train_sel, np.reshape(weight_sel, (no_selected, 1)))+b
        output = np.ravel(output)
        loss = 0.5*Paras.alpha*norm(output*self.y_train-1) #/(self.no_instances*(self.no_features+2)**2)
        if no_selected == 0:
            fitness = self.worst_fitness()
        else:
            fitness = reg_weight + loss + pen_sel

        return fitness

    def gradient_step(self, sol, l_rate):
        weight = np.copy(sol[0:self.no_features+1])
        weight = np.reshape(weight, (1, len(weight)))
        X_add_1 = np.append(self.X_train, np.ones((self.no_instances, 1)), axis=1)

        delta = np.matmul(np.matmul(weight, X_add_1.T) - np.reshape(self.y_train, (1, self.no_instances)),
                            X_add_1)
        delta = delta/np.max(abs(delta))
        weight = np.ravel(weight - l_rate*delta)
        weight = weight/np.sqrt(np.dot(weight, weight.T))

        new_sol = np.copy(sol)
        new_sol[0:self.no_features+1] = np.ravel(weight)
        return new_sol

