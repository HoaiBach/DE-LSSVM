"""
Authors: Bach Nguyen
Created: 03/06/2021
Description:
-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
"""
import os
import scipy.io
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


datasets = ['German', 'Sonar', 'Musk1', 'LSVT','WBCD', 'QsarAndrogenReceptor']
bench_methods = ['RFS', 'GFS']

# extract for DEFS
for dataset in datasets:
    method_features = dict()

    # Extract for DEFS
    method = 'embed_not-norm_100'
    no_runs = 30
    best_fold_features = []
    best_fold_acc = []
    for run in range(1, no_runs + 1):
        if os.path.exists('/local/scratch/GridResults/JADE/10Fold/Balance_accuracy/' +
                          dataset + '/' + method + '/' + str(run) + '.txt'):
            f = open('/local/scratch/GridResults/JADE/10Fold/Balance_accuracy/' +
                     dataset + '/' + method + '/' + str(run) + '.txt', 'r')
            lines = f.readlines()

            for line_idx in range(len(lines)):
                line = lines[line_idx]
                if 'Fold' in line:
                    splits = line.split(' ')
                    fold_idx = int(splits[2])-1
                    # extract features
                    line_idx = line_idx + 109
                    features = [int(ele) for ele in lines[line_idx].replace(', \n', '').split(': ')[1].split(', ')]
                    # extract acc
                    line_idx = line_idx + 8
                    acc = float(lines[line_idx].split(': ')[1])
                    # now add to best records
                    if len(best_fold_acc) < (fold_idx+1):
                        assert len(best_fold_acc) == fold_idx
                        best_fold_acc.append(acc)
                        best_fold_features.append(features)
                    else:
                        if best_fold_acc[fold_idx] < acc:
                            best_fold_acc[fold_idx] = acc
                            best_fold_features[fold_idx] = features
                        elif best_fold_acc[fold_idx] == acc and len(best_fold_features[fold_idx]) > len(features):
                            best_fold_features[fold_idx] = features
    method_features.update({'DEFS': best_fold_features})

    for method in bench_methods:
        print(method)
        best_fold_features = []

        if os.path.exists('/local/scratch/GridResults/JADE/10Fold/Balance_accuracy/' +
                          dataset + '/' + method + '.txt'):
            f = open('/local/scratch/GridResults/JADE/10Fold/Balance_accuracy/' +
                     dataset + '/' + method + '.txt', 'r')
            lines = f.readlines()

            for line_idx in range(len(lines)):
                line = lines[line_idx]
                if 'Fold' in line:
                    splits = line.split(' ')
                    fold_idx = int(splits[2])-1
                    # extract features
                    while not (lines[line_idx].startswith('Selected features')):
                        line_idx += 1
                    features = [int(ele) for ele in lines[line_idx].replace(', \n', '').split(': ')[1].split(', ')]
                    best_fold_features.append(features[: len(method_features.get('DEFS')[fold_idx])])
        method_features.update({method: best_fold_features})

    print('Done loading')
    #load data
    mat = scipy.io.loadmat('/vol/grid-solar/sgeusers/nguyenhoai2/Dataset/FSMatlab/'+dataset+'.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]

    no_folds = len(method_features.get('DEFS'))
    if no_folds > 1:
        sfold = StratifiedKFold(n_splits=no_folds, shuffle=True, random_state=1617)
        index = 0
        for train_index, test_index in sfold.split(X, y):
            X_train = X[train_index, :]
            y_train = y[train_index]
            df = pd.DataFrame(X_train)
            df['label'] = y_train
            df.head()
            cor = df.corr()
            cor = np.array(cor.values)
            cor = np.abs(cor)
            fig, axes = plt.subplots(ncols=len(method_features.keys()), figsize=(12, 4))
            for m_idx, method in enumerate(method_features.keys()):
                features = method_features.get(method)[index]
                method_cor = cor[features, :][:, features]
                ax = axes[m_idx]
                ax.set_title(method, fontsize=20, fontweight='bold')
                im = ax.matshow(method_cor)
                ax.xaxis.set_ticks_position('bottom')
                ax.tick_params(axis='both', labelsize=14)
                fig.colorbar(im, ax=ax)
            fig.tight_layout()
            plt.savefig('/run/media/nguyenhoai2/Bach/ICDM2021/Fig/FCor/'+dataset+'-'+str(index)+'.pdf'
                        , format='pdf',  bbox_inches='tight')
            plt.close()
            index += 1
    else:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1617)
        df = pd.DataFrame(X)
        df['label'] = y
        df.head()
        cor = df.corr()
        cor = np.array(cor.values)
        cor = np.abs(cor)
        fig, axes = plt.subplots(ncols=len(method_features.keys()), figsize=(12, 4))
        for m_idx, method in enumerate(method_features.keys()):
            features = method_features.get(method)[0]
            method_cor = cor[features, :][:, features]
            ax = axes[m_idx]
            ax.set_title(method, fontsize=20, fontweight='bold')
            im = ax.matshow(method_cor)
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis='both', labelsize=14)
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.savefig(
            '/run/media/nguyenhoai2/Bach/ICDM2021/Fig/FCor/' + dataset + '-' + str(0) + '.pdf'
            , format='pdf', bbox_inches='tight')
        plt.close()
