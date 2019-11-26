import numpy as np
import math
import base
from sklearn.model_selection import StratifiedKFold
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


class FS_Wrapper(Problem):

    def __init__(self, X, y, clf):
        Problem.__init__(self, minimized=True)
        self.X = X
        self.y = y
        num_class, count = np.unique(self.y, return_counts=True)
        min_class = np.min(count)
        n_splits = min(min_class, 10)
        self.skf = StratifiedKFold(n_splits=n_splits, random_state=1617)
        self.no_features = self.X.shape[1]
        self.clf = clf

    def fitness(self, sol):
        sel_idx = np.where(sol > Paras.threshold)[0]
        if len(sel_idx) == 0:
            return self.worst_fitness(), 0, self.worst_fitness()

        sum_err = 0
        fold_count = 0
        X_sel = self.X[:, sel_idx]

        for train_idx, test_idx in self.skf.split(X_sel, self.y):
            X_train, X_test = X_sel[train_idx], X_sel[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            self.clf.fit(X_train, y_train)
            err = 1 - balanced_accuracy_score(y_test, self.clf.predict(X_test))
            sum_err += err
            fold_count = fold_count+1
        ave_err = sum_err/fold_count
        sel_ratio = float(len(sel_idx))/len(sol)
        fitness = Paras.w_wrapper*ave_err + (1.0-Paras.w_wrapper)*sel_ratio
        return fitness, sel_ratio, ave_err


class FS_LSSVM(Problem):

    def __init__(self, X, y):
        Problem.__init__(self, minimized=True)
        self.X = X
        self.y = np.copy(y)
        self.no_instances, self.no_features = X.shape[0], X.shape[1]
        unique_y, count = np.unique(self.y, return_counts=True)
        self.class_weight = np.array([1.0,]*self.no_instances)
        assert len(unique_y) == 2
        for y_value, y_count in zip(unique_y, count):
            indices = np.where(self.y == y_value)[0]
            self.class_weight[indices] = 1.0/y_count

    def regularization(self, w):
        if Paras.reg == 'l2':
            return np.sqrt(np.sum(w**2))
        elif Paras.reg == 'l1':
            return np.sum(np.abs(w))
        elif Paras.reg == 'l0':
            return np.sum(np.abs(w) > 0)/float(self.no_features)
        else:
            return np.sqrt(np.sum(w**2))

    def loss(self, output):
        if Paras.loss == 'H':
            return np.sum(np.maximum(1 - output * self.y, 0))
        elif Paras.loss == 'B':
            return len(np.where(output*self.y <= 0)[0])/float(self.no_instances)
        else:
            return np.sum(np.maximum(1 - output * self.y, 0))

    def fitness(self, sol_ori):
        sol = np.array(sol_ori)
        weight = sol[0:self.no_features]
        b = sol[self.no_features:self.no_features+1]
        mask = sol[self.no_features+1:]
        selected_index = np.where(mask > Paras.threshold)[0]

        X_sel = self.X[:, selected_index]
        weight_sel = weight[selected_index]
        no_selected = len(selected_index)

        # calculate reg
        if Paras.max_reg < 0:
            reg_weight = self.regularization(weight_sel)
        else:
            reg_weight = (self.regularization(weight_sel)-Paras.min_reg)/(Paras.max_reg-Paras.min_reg)

        # calculate loss part
        output = np.dot(X_sel, np.reshape(weight_sel, (no_selected, 1)))+b
        output = np.ravel(output)
        if Paras.max_loss < 0:
            loss = self.loss(output)
        else:
            loss = (self.loss(output) - Paras.min_loss) / (Paras.max_loss - Paras.min_loss)

        if no_selected == 0:
            fitness = self.worst_fitness()
        else:
            fitness = reg_weight + Paras.alpha * loss

        return fitness, reg_weight, loss

    def fitness_binary(self, sol_ori):
        sol = np.array(sol_ori)
        weight = sol[0:self.no_features]
        b = sol[self.no_features:self.no_features+1]
        mask = sol[self.no_features+1:]
        selected_index = np.where(mask > Paras.threshold)[0]

        X_sel = self.X[:, selected_index]
        weight_sel = weight[selected_index]
        no_selected = len(selected_index)

        # calculate reg
        if Paras.max_reg < 0:
            reg_weight = np.sum(np.abs(weight_sel))
        else:
            reg_weight = (np.sum(np.abs(weight_sel))-Paras.min_reg)/(Paras.max_reg-Paras.min_reg)

        # calculate loss part
        output = np.dot(X_sel, np.reshape(weight_sel, (no_selected, 1)))+b
        output = np.ravel(output)
        consis = output * self.y
        mark = np.ones((len(consis),))
        mark[consis > 0] = 0.0
        mark[consis <= 0] = 1.0
        if Paras.max_loss < 0:
            loss = np.sum(self.class_weight*mark)
        else:
            loss = (np.sum(self.class_weight*mark)-Paras.min_loss)/(Paras.max_loss-Paras.min_loss)

        if no_selected == 0:
            fitness = self.worst_fitness()
        else:
            fitness = reg_weight + Paras.alpha * loss

        return fitness, reg_weight, loss

    def fitness_balance(self, sol_ori):
        sol = np.array(sol_ori)
        weight = sol[0:self.no_features]
        b = sol[self.no_features:self.no_features+1]
        mask = sol[self.no_features+1:]
        selected_index = np.where(mask > Paras.threshold)[0]

        X_sel = self.X[:, selected_index]
        weight_sel = weight[selected_index]
        no_selected = len(selected_index)

        # calculate reg
        if Paras.max_reg < 0:
            reg_weight = np.sum(np.abs(weight_sel))
        else:
            reg_weight = (np.sum(np.abs(weight_sel))-Paras.min_reg)/(Paras.max_reg-Paras.min_reg)

        # calculate loss part
        output = np.dot(X_sel, np.reshape(weight_sel, (no_selected, 1)))+b
        output = np.ravel(output)
        if Paras.max_loss < 0:
            loss = np.sum(self.class_weight*np.maximum(1 - output * self.y, 0))
        else:
            loss = (np.sum(self.class_weight*np.maximum(1 - output * self.y, 0)) - Paras.min_loss) / (Paras.max_loss - Paras.min_loss)

        if no_selected == 0:
            fitness = self.worst_fitness()
        else:
            fitness = reg_weight + Paras.alpha * loss

        return fitness, reg_weight, loss

    def gradient_step(self, sol, l_rate):
        weight = np.copy(sol[0:self.no_features+1])
        weight = np.reshape(weight, (1, len(weight)))
        X_add_1 = np.append(self.X, np.ones((self.no_instances, 1)), axis=1)

        delta = np.matmul(np.matmul(weight, X_add_1.T) - np.reshape(self.y, (1, self.no_instances)),
                          X_add_1)
        delta = delta/np.max(abs(delta))
        weight = np.ravel(weight - l_rate*delta)
        weight = weight/np.sqrt(np.dot(weight, weight.T))

        new_sol = np.copy(sol)
        new_sol[0:self.no_features+1] = np.ravel(weight)
        return new_sol

    def generate_candidate(self, sol):
        mask = sol[self.no_features + 1:]
        selected_index = np.where(mask > Paras.threshold)[0]
        if len(selected_index) == 0:
            selected_index = np.array([np.random.randint(0, self.no_features)])

        tmp_clf = svm.LinearSVC(random_state=1617, C=1.0, penalty='l1', dual=False)
        X_train_sel = self.X[:, selected_index]
        tmp_clf.fit(X_train_sel, self.y)
        coef = tmp_clf.coef_[0]
        intercept = tmp_clf.intercept_[0]
        assert len(selected_index) == len(coef)

        new_sol = np.zeros(len(sol))
        new_sol[self.no_features] = intercept
        for f_idx, f_weight in zip(selected_index, coef):
            new_sol[f_idx] = f_weight
            if f_weight != 0:
                new_sol[self.no_features+f_idx+1] = 1.0
        return new_sol

    def generate_candidate_with_nf(self, no_sel):
        selected_index = np.random.choice(self.no_features, size=no_sel, replace=False)
        if len(selected_index) == 0:
            selected_index = np.array([np.random.randint(0, self.no_features)])

        tmp_clf = svm.LinearSVC(random_state=1617, C=1.0, penalty='l1', dual=False)
        X_train_sel = self.X[:, selected_index]
        tmp_clf.fit(X_train_sel, self.y)
        coef = tmp_clf.coef_[0]
        intercept = tmp_clf.intercept_[0]
        assert len(selected_index) == len(coef)

        # tmp_clf = svm.LinearSVC(random_state=1617, C=1.0)
        # X_train_sel = self.X[:, selected_index]
        # tmp_clf.fit(X_train_sel, self.y)
        # coef = tmp_clf.coef_[0]
        # intercept = tmp_clf.intercept_[0]
        # assert len(selected_index) == len(coef)

        new_sol = np.zeros(2*self.no_features+1)
        new_sol[self.no_features] = intercept
        for f_idx, f_weight in zip(selected_index, coef):
            new_sol[f_idx] = f_weight
            if f_weight != 0:
                new_sol[self.no_features+f_idx+1] = 1.0
        return new_sol
