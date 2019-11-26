import numpy as np
import os
import pandas as pd
from collections import OrderedDict
import base



def find_g0(a):
    b = np.copy(a)
    return b[np.where(b > 0)[0]]

# datasets = ['German', 'WBCD', 'Lung', 'Sonar', 'Musk1', 'LSVT', 'Semeion', 'Madelon', 'Ovarian','Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia',
#          'Prostate']


datasets = ['Australian'
    , 'Spect', 'German', 'WBCD', 'Ionosphere', 'Lung', 'Sonar', 'Hillvalley', 'Parkinson',
            'Musk1', 'LSVT', 'Madelon',  'Semeion', 'Ovarian','Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia', 'arcene',
         'Prostate', 'GLI-85']

de_methods = [
           #  'embed_not-norm_1_interval_B_l0',
           #  'embed_not-norm_10_interval_B_l0',
           # 'embed_not-norm_100_interval_B_l0',
           # 'embed_not-norm_1_interval_B_l1',
           # 'embed_not-norm_10_interval_B_l1',
           # 'embed_not-norm_100_interval_B_l1',
           # 'embed_not-norm_1_interval_B_l2',
           # 'embed_not-norm_10_interval_B_l2',
           # 'embed_not-norm_100_interval_B_l2',
           # 'embed_not-norm_1_interval_H_l0',
           # 'embed_not-norm_10_interval_H_l0',
           # 'embed_not-norm_100_interval_H_l0',
           # 'embed_not-norm_1',
           # 'embed_not-norm_10',
           'embed_not-norm_100',
           #  'embed_not-norm_1_interval_B_l0_v2',
           #  'embed_not-norm_10_interval_B_l0_v2',
           # 'embed_not-norm_100_interval_B_l0_v2',
    'LOEC'
    #        'embed_not-norm_1_interval_H_l2',
    #        'embed_not-norm_10_interval_H_l2',
    #        'embed_not-norm_100_interval_H_l2',
           ]
de_methods_short = [
    # 'B_l0_1', 'B_l0_10', 'B_l0_100',
    #             'B_l1_1', 'B_l1_10', 'B_l1_100',
    #             'B_l2_1', 'B_l2_10', 'B_l2_100',
    #             'H_l0_1', 'H_l0_10', 'H_l0_100',
                # 'H_l1_1', 'H_l1_10',
                'H_l1_100',
# 'B_l0_1_v2', 'B_l0_10_v2', 'B_l0_100_v2',
    'LOEC'
    #             'H_l2_1', 'H_l2_10', 'H_l2_100'
                 ]
tr_methods = ['CFS', 'GFS', 'mRMR', 'reliefF', 'RFS']
master = 'H_l1_100'
no_runs = 30

knn_accs = OrderedDict([('Datasets', [])])
knn_accs_sig = OrderedDict([('Datasets', [])])
svm_accs = OrderedDict([('Datasets', [])])
svm_accs_sig = OrderedDict([('Datasets', [])])
nfs = OrderedDict([('Datasets', [])])

knn_accs.update({'Full': []})
svm_accs.update({'Full': []})
knn_accs_sig.update({'Full': []})
svm_accs_sig.update({'Full': []})
for method_short in de_methods_short+tr_methods:
    knn_accs.update({method_short: []})
    svm_accs.update({method_short: []})
    knn_accs_sig.update({method_short: []})
    svm_accs_sig.update({method_short: []})
    nfs.update({method_short: []})

for data_idx, dataset in enumerate(datasets):
    knn_accs['Datasets'].append(dataset)
    svm_accs['Datasets'].append(dataset)
    knn_accs_sig['Datasets'].append(dataset)
    svm_accs_sig['Datasets'].append(dataset)
    nfs['Datasets'].append(dataset)

    knn_full = []
    svm_full = []

    for m_idx, method in enumerate(de_methods):
        method_short = de_methods_short[m_idx]
        knn_sel = []
        svm_sel = []
        n_sel = []

        for run in range(1, no_runs + 1):
            if os.path.exists('/local/scratch/GridResults/FeatureSelection/10Fold/Balance_accuracy/' +
                              dataset + '/' + method + '/' + str(run) + '.txt'):
                f = open('/local/scratch/GridResults/FeatureSelection/10Fold/Balance_accuracy/' +
                         dataset + '/' + method + '/' + str(run) + '.txt', 'r')
                lines = f.readlines()

                l_idx = 0
                for line in lines:
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

        knn_accs[method_short].append(knn_sel)
        svm_accs[method_short].append(svm_sel)
        nfs[method_short].append(n_sel)
        if (len(knn_accs['Full'])-1)<data_idx:
            knn_accs['Full'].append(knn_full)
            svm_accs['Full'].append(svm_full)

    for m_idx, method in enumerate(tr_methods):
        method_short = tr_methods[m_idx]
        knn_sel = []
        svm_sel = []
        n_sel = []

        if os.path.exists('/home/nguyenhoai2/Grid/results/FeatureSelection/10Fold/Balance_accuracy/' +
                          dataset + '/' + method + '.txt'):
            f = open('/home/nguyenhoai2/Grid/results/FeatureSelection/10Fold/Balance_accuracy/' +
                     dataset + '/' + method + '.txt', 'r')
            lines = f.readlines()

            l_idx = 0
            for line in lines:
                if 'Final results' in line:
                    break
                else:
                    l_idx += 1

            svm_sel.append(float(lines[l_idx + 2].split(': ')[1]))
            knn_sel.append(float(lines[l_idx + 4].split(': ')[1]))
            n_sel.append(float(lines[l_idx - 2].split(': ')[1]))
        else:
            svm_sel.append(0)
            knn_sel.append(0)
            n_sel.append(0)

        knn_accs[method_short].append(knn_sel*no_runs)
        svm_accs[method_short].append(svm_sel*no_runs)
        nfs[method_short].append(n_sel*no_runs)

    # significance test
    master_result_knn = knn_accs[master][data_idx]
    master_result_svm = svm_accs[master][data_idx]
    for method in de_methods_short+tr_methods+['Full']:
        method_result_knn = knn_accs[method][data_idx]
        method_result_svm = svm_accs[method][data_idx]
        sig_result = base.wilcoxon_test(master_result_knn, method_result_knn, minimized=False)
        knn_accs_sig[method].append(sig_result)
        sig_result = base.wilcoxon_test(master_result_svm, method_result_svm, minimized=False)
        svm_accs_sig[method].append(sig_result)

for method in de_methods_short+tr_methods+['Full']:
    tmp_ave = []
    for arr_acc in knn_accs[method]:
        tmp_ave.append(np.mean(find_g0(arr_acc)))
    knn_accs[method] = tmp_ave
    tmp_ave = []
    for arr_acc in svm_accs[method]:
        tmp_ave.append(np.mean(find_g0(arr_acc)))
    svm_accs[method] = tmp_ave

for method in de_methods_short + tr_methods:
    tmp_ave = []
    for arr_nf in nfs[method]:
        tmp_ave.append(np.mean(find_g0(arr_nf)))
    nfs[method] = tmp_ave

df_svm = pd.DataFrame(svm_accs).round(4)
# print(df_svm.to_csv(index=False))
print(df_svm.to_latex(index=False))
df_svm_sig = pd.DataFrame(svm_accs_sig)
print(df_svm_sig.to_latex(index=False))

df_knn = pd.DataFrame(knn_accs).round(4)
print(df_knn.to_latex(index=False))
# print(df_knn.to_csv(index=False))
df_knn_sig = pd.DataFrame(knn_accs_sig)
print(df_knn_sig.to_latex(index=False))

df_nf = pd.DataFrame(nfs).round(4)
print(df_nf.to_latex(index=False))
# print(df_nf.to_csv(index=False))


