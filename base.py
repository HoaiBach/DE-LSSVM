import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from scipy.stats import wilcoxon


def jade_mutant(x_i, x_b, x_r1, x_r2, F):
    mutant = x_i + F*(x_b-x_i) + F*(x_r1-x_r2)
    # for idx, value in enumerate(mutant):
    #     if value > maxpos:
    #         mutant[idx] = (maxpos+x_i[idx])/2
    #     if value < minpos:
    #         mutant[idx] = (minpos+x_i[idx])/2
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
            trial[idx] = (maxpos[idx]+x_i[idx])/2
            # trial[idx] = maxpos
        if value < minpos[idx]:
            trial[idx] = (minpos[idx]+x_i[idx])/2
            # trial[idx] = minpos
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
    if np.sum(np.abs(np.array(mine)-np.array(enemy))) == 0:
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
