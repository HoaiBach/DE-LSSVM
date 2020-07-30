import numpy as np
import os
import pandas as pd
from collections import OrderedDict
import Base
import tables
import scipy
import nonparametric_tests as nontest
import scikit_posthocs as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def find_g0(a):
    b = np.copy(a)
    return b[np.where(b > 0)[0]]

def twof(a):
    return format(a, '.2f')

# Helper functions for performing the statistical tests
def generate_scores(method, method_args, data, labels):
    pairwise_scores = method(data, **method_args) # Matrix for all pairwise comaprisons
    pairwise_scores.set_axis(labels, axis='columns', inplace=True) # Label the cols
    pairwise_scores.set_axis(labels, axis='rows', inplace=True) # Label the rows, note: same label as pairwise combinations
    return pairwise_scores

def plot(scores, ax):
    # Pretty plot of significance
    heatmap_args = {'linewidths': 1, 'linecolor': '0.5', 'square': True,
                    'cbar_ax_bbox': [0.82, 0.35, 0.04, 0.3]}

    sp.sign_plot(scores, ax=ax, **heatmap_args)


# datasets = ['German', 'WBCD', 'Lung', 'Sonar', 'Musk1', 'LSVT', 'Semeion', 'Madelon', 'Ovarian','Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia',
#          'Prostate']


datasets = ['Australian'
    , 'Spect', 'German', 'WBCD', 'Ionosphere', 'Lung', 'Sonar', 'Hillvalley', 'Parkinson',
            'Musk1', 'LSVT', 'Madelon',  'Semeion', 'Ovarian','Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia', 'arcene',
         'Prostate', 'GLI-85']

datasets = [ 'Parkinson','German', 'WBCD', 'Sonar',
            'Musk1', 'LSVT', 'Madelon','Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia',
         'Prostate', 'Ovarian']


de_methods = [
           'embed_not-norm_1', 'embed_not-norm_10', 'embed_not-norm_100', 'embed_not-norm_1000'
           ]
de_methods_short = [
                '0.001', '0.01', '0.1', '1.0'
                 ]
no_runs = 30

knn_accs = OrderedDict([('Datasets', [])])
svm_accs = OrderedDict([('Datasets', [])])
nfs = OrderedDict([('Datasets', [])])

for method_short in de_methods_short:
    knn_accs.update({method_short: []})
    svm_accs.update({method_short: []})
    nfs.update({method_short: []})

for data_idx, dataset in enumerate(datasets):
    knn_accs['Datasets'].append(dataset)
    svm_accs['Datasets'].append(dataset)
    nfs['Datasets'].append(dataset)

    # read the number of features for full
    mat = scipy.io.loadmat('/home/nguyenhoai2/Grid/data/FSMathlab/'+dataset+'.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    full_ni, full_nf = X.shape
    unique_classes = np.unique(y)
    y[y == unique_classes[0]] = 0
    y[y == unique_classes[1]] = 1
    full_clp = (float(np.sum(y == 1)))/full_ni
    full_cln = (float(np.sum(y == 0)))/full_ni

    for m_idx, method in enumerate(de_methods):
        method_short = de_methods_short[m_idx]
        knn_sel = []
        svm_sel = []
        n_sel = []
        de_time = []

        for run in range(1, no_runs + 1):
            if os.path.exists('/local/scratch/GridResults/FeatureSelection/10Fold/Balance_accuracy/' +
                              dataset + '/' + method + '/' + str(run) + '.txt'):
                f = open('/local/scratch/GridResults/FeatureSelection/10Fold/Balance_accuracy/' +
                         dataset + '/' + method + '/' + str(run) + '.txt', 'r')
                lines = f.readlines()

                l_idx = 0
                for line in lines:
                    # extract time
                    if 'Time:' in line:
                        de_time.append(float(line.split(': ')[1]))
                    if 'Final results' in line:
                        break
                    else:
                        l_idx += 1

                svm_sel.append(float(lines[l_idx+2].split(': ')[1]))
                knn_sel.append(float(lines[l_idx+4].split(': ')[1]))
                if (l_idx+5) < len(lines):
                    n_sel.append(float(lines[l_idx+5].split(': ')[1]))
                else:
                    n_sel.append(float(lines[l_idx-2].split(': ')[1]))
            else:
                svm_sel.append(0)
                knn_sel.append(0)
                n_sel.append(0)

        knn_accs[method_short].append(knn_sel)
        svm_accs[method_short].append(svm_sel)
        nfs[method_short].append(n_sel)

# perform friedman test

sns.set(font_scale=2.0)
friedman_results = []
for dataset_idx, dataset in enumerate(datasets):
    knn_result = []
    svm_result = []
    for method in de_methods_short:
        knn_result.append(1.0-np.array(knn_accs[method][dataset_idx]))
        svm_result.append(1.0-np.array(svm_accs[method][dataset_idx]))
    # _, _, knn_rank, knn_pivots = nontest.friedman_test(*knn_result)
    # _, _, svm_rank, svm_pivots = nontest.friedman_test(*svm_result)
    # knn_pivots_dict = {key: knn_pivots[i] for i, key in enumerate(de_methods_short)}
    # knn_compare, _, _, knn_pvalues = nontest.nemenyi_multitest(knn_pivots_dict)
    # svm_pivots_dict = {key: svm_pivots[i] for i, key in enumerate(de_methods_short)}
    # _, _, _, svm_pvalues = nontest.nemenyi_multitest(svm_pivots_dict)
    fig, ax = plt.subplots()
    plt.title(dataset, fontsize=24, fontweight='bold')
    df_knn = pd.DataFrame(np.array(knn_result).T, columns=de_methods_short)
    df_svm = pd.DataFrame(np.array(svm_result).T, columns=de_methods_short)
    nemenyi_scores_svm = generate_scores(sp.posthoc_nemenyi_friedman, {}, df_svm, de_methods_short)
    plot(nemenyi_scores_svm, ax)
    # plt.tight_layout()
    plt.savefig('Fig/beta/'+dataset+'.pdf', format='pdf', bbox_inches='tight')
    plt.close()



