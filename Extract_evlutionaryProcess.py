import numpy as np
import os
import pandas as pd
from collections import OrderedDict
import base
import tables
import scipy
import nonparametric_tests as nontest
import matplotlib
import matplotlib.pyplot as plt


def find_g0(a):
    b = np.copy(a)
    return b[np.where(b > 0)[0]]

def twof(a):
    return format(a, '.2f')

# datasets = ['German', 'WBCD', 'Lung', 'Sonar', 'Musk1', 'LSVT', 'Semeion', 'Madelon', 'Ovarian','Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia',
#          'Prostate']


datasets = ['Australian'
    , 'Spect', 'German', 'WBCD', 'Ionosphere', 'Lung', 'Sonar', 'Hillvalley', 'Parkinson',
            'Musk1', 'LSVT', 'Madelon',  'Semeion', 'Ovarian','Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia', 'arcene',
         'Prostate', 'GLI-85']

datasets = [ 'Parkinson','German', 'WBCD', 'Sonar',
            'Musk1', 'LSVT', 'Madelon','Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia',
         'Prostate', 'Ovarian']

datasets = [ 'German', 'Musk1', 'DLBCL', 'Leukemia']


de_methods = ['embed_not-norm_100', 'embed_not-norm_100_random']
de_methods_short = ['Ratio', 'Random']
no_runs = 30
no_iter = 100

knn_accs = OrderedDict([('Datasets', [])])
svm_accs = OrderedDict([('Datasets', [])])
eps = OrderedDict([('Datasets',[])])
nfs = OrderedDict([('Datasets', [])])
data_info = OrderedDict([('Datasets', []), ('nf', []), ('ni', []), ('clp', []), ('cln', [])])


for method_short in de_methods_short:
    knn_accs.update({method_short: []})
    svm_accs.update({method_short: []})
    eps.update({method_short: []})
    nfs.update({method_short: []})

for data_idx, dataset in enumerate(datasets):
    knn_accs['Datasets'].append(dataset)
    svm_accs['Datasets'].append(dataset)
    eps['Datasets'].append(dataset)
    nfs['Datasets'].append(dataset)
    data_info['Datasets'].append(dataset)

    knn_full = []
    svm_full = []
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
    data_info['nf'].append(full_nf)
    data_info['ni'].append(full_ni)
    data_info['clp'].append(full_clp)
    data_info['cln'].append(full_cln)

    for m_idx, method in enumerate(de_methods):
        method_short = de_methods_short[m_idx]
        knn_sel = []
        svm_sel = []
        n_sel = []
        de_time = []
        ep = [0,]*no_iter
        no_ep_repeat = 0

        for run in range(1, no_runs + 1):
            if os.path.exists('/local/scratch/GridResults/FeatureSelection/10Fold/Balance_accuracy/' +
                              dataset + '/' + method + '/' + str(run) + '.txt'):
                f = open('/local/scratch/GridResults/FeatureSelection/10Fold/Balance_accuracy/' +
                         dataset + '/' + method + '/' + str(run) + '.txt', 'r')
                lines = f.readlines()

                l_idx = 0
                for line in lines:
                    if 'Fold' in line:
                        no_ep_repeat += 1
                    if 'Iteration' in line:
                        index_str, fit_str = line.split(': ')
                        iter_index = int(index_str.split(' ')[1])
                        if iter_index < no_iter:
                            fit = float(fit_str.split(',')[0])
                            ep[iter_index] = ep[iter_index]+fit
                    # extract time
                    if 'Time:' in line:
                        de_time.append(float(line.split(': ')[1]))
                    if 'Final results' in line:
                        break
                    else:
                        l_idx += 1

                if len(knn_full) < no_runs:
                    svm_full.append(float(lines[l_idx+1].split(': ')[1]))
                    knn_full.append(float(lines[l_idx+3].split(': ')[1]))

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

        knn_accs[method_short].append(np.mean(knn_sel))
        svm_accs[method_short].append(np.mean(svm_sel))
        nfs[method_short].append(np.mean(n_sel)/float(full_nf))
        ep = np.array(ep)/no_ep_repeat/1000
        eps[method_short].append(ep)

# draw nf bar
draw_method_nf = []
for method_idx, method in enumerate(de_methods_short):
    nf = []
    for dataset_idx, dataset in enumerate(datasets):
        nf.append(nfs[method][dataset_idx])
    draw_method_nf.append(nf)

x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects = []
mid = len(de_methods_short)/2
for method_idx, method in enumerate(de_methods_short):
    rect = ax.bar(x + (method_idx-mid+0.5)*width, draw_method_nf[method_idx], width, label=method)
    rects.append(rect)

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Feature ratio')
ax.set_title('Selected feature ratio')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

for rect in rects:
    autolabel(rect, ax)
fig.tight_layout()
plt.savefig('Fig/vsRandom/Nf', format='pdf')
plt.close()

# draw acc bar
draw_method_acc = []
for method_idx, method in enumerate(de_methods_short):
    acc = []
    for dataset_idx, dataset in enumerate(datasets):
        acc.append(svm_accs[method][dataset_idx])
    draw_method_acc.append(acc)

x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

fig_acc, ax_acc = plt.subplots()
rects_acc = []
mid = len(de_methods_short)/2
for method_idx, method in enumerate(de_methods_short):
    rect_acc = ax_acc.bar(x + (method_idx-mid+0.5)*width, draw_method_acc[method_idx], width, label=method)
    rects_acc.append(rect_acc)

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Feature ratio')
ax_acc.set_title('SVM accuracy')
ax_acc.set_xticks(x)
ax_acc.set_xticklabels(datasets)
ax_acc.legend()

for rect in rects_acc:
    autolabel(rect, ax_acc)
fig_acc.tight_layout()
plt.savefig('Fig/vsRandom/SVMacc', format='pdf')
plt.close()

# multiple line plot
lstys = ['-', '--', '-.']
for dataset_idx, dataset in enumerate(datasets):
    iterations = np.arange(no_iter)
    for method_idx, method in enumerate(de_methods_short):
        fitness = eps[method][dataset_idx]
        plt.plot(iterations, fitness, lstys[method_idx], label=method)
    plt.legend(loc='upper right')
    plt.title(dataset, fontsize=16, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.savefig('Fig/vsRandom/'+dataset, format='pdf')
    plt.close()
