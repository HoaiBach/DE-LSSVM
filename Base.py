import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from scipy.stats import wilcoxon
import scipy.io
from sklearn.model_selection import StratifiedKFold
import Paras


def jade_mutant(x_i, x_b, x_r1, x_r2, F):
    mutant = x_i + F * (x_b - x_i) + F * (x_r1 - x_r2)
    return mutant


def jade_crossover(x_i, v, CR, minpos, maxpos):
    cr_rnd = np.random.rand(x_i.shape[0])
    j_rnd = np.random.randint(x_i.shape[0])
    mask = cr_rnd < CR
    mask[j_rnd] = True
    trial = np.copy(x_i)
    trial[mask] = v[mask]
    for idx, value in enumerate(trial):
        if value > maxpos[idx]:
            trial[idx] = (maxpos[idx] + x_i[idx]) / 2
        if value < minpos[idx]:
            trial[idx] = (minpos[idx] + x_i[idx]) / 2
    return trial


def kernel_matrix(X1, X2=None, kernel='linear'):
    if kernel == 'linear':
        if X2 is None:
            return linear_kernel(X1, X1)
        else:
            return linear_kernel(X1, X2)
    elif kernel == 'poly':
        if X2 is None:
            return polynomial_kernel(X1, X1, degree=3)
        else:
            return polynomial_kernel(X1, X2, degree=3)
    elif kernel == 'rbf':
        if X2 is None:
            return rbf_kernel(X1, X1, gamma=0.5)
        else:
            return rbf_kernel(X1, X2, gamma=0.5)
    else:
        raise Exception('Kernel %s is not defined!!' % kernel)


def wilcoxon_test(mine, enemy, minimized):
    # 0 for mine == enemy, 1 for mine > enemy, -1 for mine < enemy
    if np.sum(np.abs(np.array(mine) - np.array(enemy))) == 0:
        return '='
    compare_result = 0
    _, pvalue = wilcoxon(mine, enemy, alternative='two-sided')
    if pvalue < 0.05:
        _, pvalue = wilcoxon(mine, enemy, alternative='greater')
        if pvalue < 0.05:
            compare_result = 1
        else:
            _, pvalue = wilcoxon(mine, enemy, alternative='less')
            if pvalue < 0.05:
                compare_result = -1
    if compare_result == 1:
        return '-' if minimized else '+'
    elif compare_result == -1:
        return '+' if minimized else '-'
    else:
        return '='


def normalise_data(X_train, X_test):
    '''
    normalize so all feature values in the range -1 and 1
    :param X_train:
    :param X_test:
    :return:
    '''
    X_min = np.min(X_train, axis=0)
    X_max = np.max(X_train, axis=0)
    non_dup = np.where(X_min != X_max)[0]
    X_min = X_min[non_dup]
    X_max = X_max[non_dup]
    X_train = X_train[:, non_dup]
    X_test = X_test[:, non_dup]
    X_train_norm = 2 * (X_train - X_min) / (X_max - X_min) - 1
    X_test_norm = 2 * (X_test - X_min) / (X_max - X_min) - 1
    return X_train_norm, X_test_norm


def load_folds(dataset):
    '''
    Load the pre-divided fold
    :param dataset:
    :return: number of folds, indicies of instances in each fold
    '''
    fold_read = open(Paras.data_dir+'FSMatlab_fold/' + dataset, 'r')
    lines = fold_read.readlines()
    no_folds = int(lines[0].split(': ')[1])
    train_folds = []
    test_folds = []
    for l_idx, line in enumerate(lines):
        if 'Fold: ' in line:

            train_line = lines[l_idx + 1]
            train_indices = []
            for i_idx in train_line.split(': ')[1].split(', '):
                train_indices.append(int(i_idx))
            train_folds.append(train_indices)

            test_line = lines[l_idx + 2]
            test_indices = []
            for i_idx in test_line.split(': ')[1].split(', '):
                test_indices.append(int(i_idx))
            test_folds.append(test_indices)

    return no_folds, train_folds, test_folds


def write_folds(dataset):
    '''
    Divide dataset into at most 5 folds, and save the fold indices
    into folder
    :param dataset:
    :return: none
    '''

    # load data
    mat = scipy.io.loadmat('/home/nguyenhoai2/Grid/data/FSMathlab/' + dataset + '.mat')
    X = mat['X']  # data
    X = X.astype(float)
    y = mat['Y']  # label
    y = y[:, 0]

    # ensure that y label is either -1 or 1
    num_class, count = np.unique(y, return_counts=True)
    n_classes = np.unique(y).shape[0]
    assert (n_classes == 2)
    min_class = np.min(count)

    no_folds = min(5, min_class)
    sfold = StratifiedKFold(n_splits=no_folds, shuffle=True, random_state=1617)
    fold_idx = 1

    fold_output = open('/home/nguyenhoai2/Grid/data/FSMathlab_fold/' + dataset, 'w')
    fold_output.write('No folds: %d\n' % no_folds)
    for train_index, test_index in sfold.split(X, y):
        fold_output.write('Fold: %d\n' % fold_idx)
        fold_idx += 1
        fold_output.write('Train: ')
        for idx_idx, idx in enumerate(train_index):
            fold_output.write('%d, ' % idx if idx_idx < len(train_index) - 1 else '%d\n' % idx)
        fold_output.write('Test: ')
        for idx_idx, idx in enumerate(test_index):
            fold_output.write('%d, ' % idx if idx_idx < len(test_index) - 1 else '%d\n' % idx)
    fold_output.close()
