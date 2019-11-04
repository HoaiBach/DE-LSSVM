import numpy as np
import os
import pandas as pd
from collections import OrderedDict

datasets = ['Australian', 'Spect', 'German', 'WBCD', 'Ionosphere', 'Lung', 'Sonar', 'Hillvalley', 'Parkinson',
            'Musk1', 'LSVT', 'Madelon', 'Ovarian', 'Semeion', 'Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia', 'arcene',
         'Prostate', 'GLI-85']
alphas = [1, 10, 100, 1000, 10000]
methods = []
methods_short = []
for alpha in alphas:
    methods.append('embed_norm_'+str(alpha))
    methods_short.append('en_'+str(alpha))
    # methods.append('embed_not-norm_'+str(alpha))
    # methods_short.append('enn_'+str(alpha))
no_runs = 30

knn_accs = OrderedDict([('Datasets', [])])
svm_accs = OrderedDict([('Datasets', [])])
nfs = OrderedDict([('Datasets', [])])

knn_accs.update({'Full': []})
svm_accs.update({'Full': []})
for method_short in methods_short:
    knn_accs.update({method_short: []})
    svm_accs.update({method_short: []})
    nfs.update({method_short: []})

for dataset in datasets:
    knn_accs['Datasets'].append(dataset)
    svm_accs['Datasets'].append(dataset)
    nfs['Datasets'].append(dataset)

    knn_full = []
    svm_full = []

    for m_idx, method in enumerate(methods):
        method_short = methods_short[m_idx]
        knn_sel = []
        svm_sel = []
        n_sel = []

        for run in range(1, no_runs + 1):
            if os.path.exists('/home/nguyenhoai2/Grid/results/FeatureSelection/10Fold/Balance_accuracy/' +
                              dataset + '/' + method + '/' + str(run) + '.txt'):
                f = open('/home/nguyenhoai2/Grid/results/FeatureSelection/10Fold/Balance_accuracy/' +
                         dataset + '/' + method + '/' + str(run) + '.txt', 'r')
                lines = f.readlines()

                if len(knn_full) < no_runs:
                    knn_full.append(float(lines[636].split(': ')[1]))
                    svm_full.append(float(lines[634].split(': ')[1]))

                knn_sel.append(float(lines[637].split(': ')[1]))
                svm_sel.append(float(lines[635].split(': ')[1]))
                n_sel.append(float(lines[638].split(': ')[1]))

        knn_accs[method_short].append(np.mean(knn_sel))
        svm_accs[method_short].append(np.mean(svm_sel))
        nfs[method_short].append(np.mean(n_sel))
    knn_accs['Full'].append(np.mean(knn_full))
    svm_accs['Full'].append(np.mean(svm_full))

df_svm = pd.DataFrame(svm_accs).round(4)
# print(df_svm.to_csv(index=False))
print(df_svm.to_latex(index=False))

df_knn = pd.DataFrame(knn_accs).round(4)
print(df_knn.to_latex(index=False))
# print(df_knn.to_csv(index=False))

df_nf = pd.DataFrame(nfs).round(4)
print(df_nf.to_latex(index=False))
# print(df_nf.to_csv(index=False))

