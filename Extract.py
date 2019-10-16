import numpy as np
import os
import pandas as pd
from collections import OrderedDict

datasets = ['Australian', 'Spect', 'Parkinson', 'German', 'WBCD', 'Ionosphere', 'Lung', 'Sonar', 'Hillvalley',
            'Musk1', 'LSVT', 'Madelon' ]
alphas = [10.0, 100.0, 1000.0]
betas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
methods = []
for alpha in alphas:
    for beta in betas:
        methods.append(str(alpha)+'_'+str(beta))

knn_accs = OrderedDict([('Datasets', [])])
svm_accs = OrderedDict([('Datasets', [])])
nfs = OrderedDict([('Datasets', [])])

knn_accs.update({'Full': []})
svm_accs.update({'Full': []})
for method in methods:
    knn_accs.update({method: []})
    svm_accs.update({method: []})
    nfs.update({method: []})

for dataset in datasets:
    knn_accs['Datasets'].append(dataset)
    svm_accs['Datasets'].append(dataset)
    nfs['Datasets'].append(dataset)

    knn_full = -1
    svm_full = -1

    for method in methods:
        if os.path.exists('/home/nguyenhoai2/Grid/results/JADE/' +
                          dataset + '/' + method+'.txt'):
            f = open('/home/nguyenhoai2/Grid/results/JADE/' +
                          dataset + '/' + method+'.txt', 'r')
            lines = f.readlines()
            if knn_full < 0:
                line = lines[1040]
                knn_full = float(line.split(': ')[1])
                knn_accs['Full'].append(knn_full)
            if svm_full < 0:
                line = lines[1042]
                svm_full = float(line.split(': ')[1])
                svm_accs['Full'].append(svm_full)
            knn_sel = float(lines[1041].split(': ')[1])
            knn_accs[method].append(knn_sel)
            svm_sel = float(lines[1043].split(': ')[1])
            svm_accs[method].append(svm_sel)
            nfs[method].append(float(lines[1044].split(': ')[1]))

df_svm = pd.DataFrame(svm_accs).round(4)
print(df_svm.to_csv(index=False))
# print(df_svm.to_latex(index=False))

df_knn = pd.DataFrame(knn_accs).round(4)
# print(df_knn.to_latex(index=False))
print(df_knn.to_csv(index=False))

df_nf = pd.DataFrame(nfs).round(4)
# print(df_nf.to_latex(index=False))
print(df_nf.to_csv(index=False))

