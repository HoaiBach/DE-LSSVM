import numpy as np
import os
import pandas as pd
from collections import OrderedDict
import Base
import tables
import scipy
import nonparametric_tests as nontest


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

datasets = [ 'Parkinson','German', 'WBCD', 'Sonar',
            'Musk1', 'LSVT', 'Madelon','Colon', 'DLBCL', 'ALLAML', 'CNS']

datasets = ['German', 'WBCD', 'Sonar', 'Musk1', 'LSVT', 'QsarAndrogenReceptor', 'ALLAML', 'Prostate', 'Ovarian']
# datasets = ['Parkinson', 'German',  'WBCD', 'Sonar', 'Musk1', 'LSVT', 'Madelon', 'Colon', 'DLBCL', 'ALLAML', 'CNS', 'Leukemia' ]

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
            'LOEC',
    #        'embed_not-norm_1_interval_H_l2',
    #        'embed_not-norm_10_interval_H_l2',
    #        'embed_not-norm_100_interval_H_l2',
    # 'wrapper_90', 'wrapper_98', 'wrapper_100'
# 'wrapper_90-p', 'wrapper_98-p', 'wrapper_100-p'
#     'JADE-relief',
#     'JADE-cor'
           ]
de_methods_short = [
    # 'B_l0_1', 'B_l0_10', 'B_l0_100',
    #             'B_l1_1', 'B_l1_10', 'B_l1_100',
    #             'B_l2_1', 'B_l2_10', 'B_l2_100',
    #             'H_l0_1', 'H_l0_10', 'H_l0_100',
                # 'H_l1_1', 'H_l1_10',
                'H_l1_100',
# 'B_l0_1_v2', 'B_l0_10_v2', 'B_l0_100_v2',
    'LOEC',
    #             'H_l2_1', 'H_l2_10', 'H_l2_100'
    # 'w90', 'w98', 'w100'
    # 'J-rel',
    # 'J-cor'
                 ]
tr_methods = ['CFS', 'GFS', 'mRMR', 'reliefF', 'RFS']
tr_time_divide = [1.0, 1.0, 1.0, 1.0, 1.0] #[1.0, 21.0, 1.0, 15.0, 21.0]
master = 'H_l1_100'
no_runs = 30

knn_accs = OrderedDict([('Datasets', [])])
knn_accs_sig = OrderedDict([('Datasets', [])])
svm_accs = OrderedDict([('Datasets', [])])
svm_accs_sig = OrderedDict([('Datasets', [])])
nfs = OrderedDict([('Datasets', [])])
times = OrderedDict([('Datasets', [])])
data_info = OrderedDict([('Datasets', []), ('nf', []), ('ni', []), ('clp', []), ('cln', [])])

knn_accs.update({'Full': []})
svm_accs.update({'Full': []})
knn_accs_sig.update({'Full': []})
svm_accs_sig.update({'Full': []})
nfs.update({'Full': []})
for method_short in de_methods_short+tr_methods:
    knn_accs.update({method_short: []})
    svm_accs.update({method_short: []})
    knn_accs_sig.update({method_short: []})
    svm_accs_sig.update({method_short: []})
    nfs.update({method_short: []})
    times.update({method_short: []})

for data_idx, dataset in enumerate(datasets):
    knn_accs['Datasets'].append(dataset)
    svm_accs['Datasets'].append(dataset)
    knn_accs_sig['Datasets'].append(dataset)
    svm_accs_sig['Datasets'].append(dataset)
    nfs['Datasets'].append(dataset)
    times['Datasets'].append(dataset)
    data_info['Datasets'].append(dataset)

    knn_full = []
    svm_full = []
    # read the number of features for full
    mat = scipy.io.loadmat('/home/nguyenhoai2/Grid/data/FSMatlab/'+dataset+'.mat')
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
    nfs['Full'].append(full_nf)

    for m_idx, method in enumerate(de_methods):
        method_short = de_methods_short[m_idx]
        knn_sel = []
        svm_sel = []
        n_sel = []
        de_time = []

        for run in range(1, no_runs + 1):
            if os.path.exists('/local/scratch/GridResults/JADE/10Fold/Balance_accuracy/' +
                              dataset + '/' + method + '/' + str(run) + '.txt'):
                f = open('/local/scratch/GridResults/JADE/10Fold/Balance_accuracy/' +
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
        times[method_short].append(de_time)
        if (len(knn_accs['Full'])-1)<data_idx:
            knn_accs['Full'].append(knn_full)
            svm_accs['Full'].append(svm_full)

    for m_idx, method in enumerate(tr_methods):
        method_short = tr_methods[m_idx]
        knn_sel = []
        svm_sel = []
        n_sel = []
        tr_time = []
        div = tr_time_divide[m_idx]

        if os.path.exists('/local/scratch/GridResults/JADE/10Fold/Balance_accuracy/' +
                          dataset + '/' + method + '.txt'):
            f = open('/local/scratch/GridResults/JADE/10Fold/Balance_accuracy/' +
                     dataset + '/' + method + '.txt', 'r')
            lines = f.readlines()

            l_idx = 0
            for line in lines:
                # extract time
                if 'Time:' in line:
                    tr_time.append(float(line.split(': ')[1])/div)
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
        times[method_short].append(tr_time)

    # significance test
    master_result_knn = knn_accs[master][data_idx]
    master_result_svm = svm_accs[master][data_idx]
    for method in de_methods_short+tr_methods+['Full']:
        method_result_knn = knn_accs[method][data_idx]
        method_result_svm = svm_accs[method][data_idx]
        sig_result = Base.wilcoxon_test(master_result_knn, method_result_knn, minimized=False)
        knn_accs_sig[method].append(sig_result)
        sig_result = Base.wilcoxon_test(master_result_svm, method_result_svm, minimized=False)
        svm_accs_sig[method].append(sig_result)

# # perform friedman test
# benchmarks = ['w90', 'w98', 'w100']
# master = ['H_l1_100']
# friedman_results = []
# for dataset_idx, dataset in enumerate(datasets):
#     knn_result = []
#     svm_result = []
#     for method in benchmarks+master:
#         knn_result.append(1.0-np.array(knn_accs[method][dataset_idx]))
#         svm_result.append(1.0-np.array(svm_accs[method][dataset_idx]))
#     _, _, knn_rank, knn_pivots = nontest.friedman_test(*knn_result)
#     _, _, svm_rank, svm_pivots = nontest.friedman_test(*svm_result)
#     knn_pivots_dict = {key: knn_pivots[i] for i, key in enumerate(benchmarks+master)}
#     _, _, _, knn_pvalues = nontest.holm_test(knn_pivots_dict)
#     svm_pivots_dict = {key: svm_pivots[i] for i, key in enumerate(benchmarks + master)}
#     _, _, _, svm_pvalues = nontest.holm_test(svm_pivots_dict)
#
#     compare_result = [dataset]
#     for ben_idx, ben in enumerate(benchmarks):
#         if svm_pvalues[ben_idx] < 0.05:
#             compare_result.append(twof(svm_rank[ben_idx])+'*')
#         else:
#             compare_result.append(twof(svm_rank[ben_idx]))
#
#     compare_result.append(twof(svm_rank[len(benchmarks)]))
#
#     for ben_idx, ben in enumerate(benchmarks):
#         if knn_pvalues[ben_idx] < 0.05:
#             compare_result.append(twof(knn_rank[ben_idx])+'*')
#         else:
#             compare_result.append(twof(knn_rank[ben_idx]))
#
#     compare_result.append(twof(knn_rank[len(benchmarks)]))
#
#     friedman_results.append(compare_result)
#
# header = [['Dataset', '', '', '', 'SVM', '', '', '', 'KNN'], ['', 'w90', 'w98', 'w100', 'DEEFS','w90', 'w98', 'w100', 'DEEFS']]
# data = header+friedman_results
# tabular = tables.Tabular(data)
# table = tables.Table(tabular)
# table.set_caption('Friedmen test: Comparing with wrapper.')
# table.set_label('tb:friedWrapper')
# print(table)

# perform friedman test, filter
benchmarks = ['CFS', 'mRMR', 'reliefF']
master = ['H_l1_100']
knn_results = []
friedman_results = []
for dataset_idx, dataset in enumerate(datasets):
    knn_result = []
    svm_result = []
    for method in benchmarks+master:
        knn_result.append(1.0-np.array(knn_accs[method][dataset_idx]))
        svm_result.append(1.0-np.array(svm_accs[method][dataset_idx]))
    _, _, knn_rank, knn_pivots = nontest.friedman_test(*knn_result)
    _, _, svm_rank, svm_pivots = nontest.friedman_test(*svm_result)
    knn_pivots_dict = {key: knn_pivots[i] for i, key in enumerate(benchmarks+master)}
    _, _, _, knn_pvalues = nontest.holm_test(knn_pivots_dict)
    svm_pivots_dict = {key: svm_pivots[i] for i, key in enumerate(benchmarks + master)}
    _, _, _, svm_pvalues = nontest.holm_test(svm_pivots_dict)

    compare_result = [dataset]
    for ben_idx, ben in enumerate(benchmarks):
        if svm_pvalues[ben_idx] < 0.05:
            compare_result.append(twof(svm_rank[ben_idx])+'*')
        else:
            compare_result.append(twof(svm_rank[ben_idx]))

    compare_result.append(twof(svm_rank[len(benchmarks)]))

    for ben_idx, ben in enumerate(benchmarks):
        if knn_pvalues[ben_idx] < 0.05:
            compare_result.append(twof(knn_rank[ben_idx])+'*')
        else:
            compare_result.append(twof(knn_rank[ben_idx]))

    compare_result.append(twof(knn_rank[len(benchmarks)]))

    friedman_results.append(compare_result)

header = [['Dataset', '', '', '', 'SVM', '', '', '', 'KNN'], ['','CFS', 'mRMR', 'reliefF', 'DEEFS',
                                                              'CFS', 'mRMR', 'reliefF', 'DEEFS']]
data = header+friedman_results
tabular = tables.Tabular(data)
table = tables.Table(tabular)
table.set_caption('Friedmen test: Comparing with filter.')
table.set_label('tb:friedFilter')
print(table)

# perform friedman test, embeded
benchmarks = ['RFS', 'GFS', 'LOEC']
master = ['H_l1_100']
knn_results = []
friedman_results = []
for dataset_idx, dataset in enumerate(datasets):
    knn_result = []
    svm_result = []
    for method in benchmarks+master:
        knn_result.append(1.0-np.array(knn_accs[method][dataset_idx]))
        svm_result.append(1.0-np.array(svm_accs[method][dataset_idx]))
    _, _, knn_rank, knn_pivots = nontest.friedman_test(*knn_result)
    _, _, svm_rank, svm_pivots = nontest.friedman_test(*svm_result)
    knn_pivots_dict = {key: knn_pivots[i] for i, key in enumerate(benchmarks+master)}
    _, _, _, knn_pvalues = nontest.holm_test(knn_pivots_dict)
    svm_pivots_dict = {key: svm_pivots[i] for i, key in enumerate(benchmarks + master)}
    _, _, _, svm_pvalues = nontest.holm_test(svm_pivots_dict)

    compare_result = [dataset]
    for ben_idx, ben in enumerate(benchmarks):
        if svm_pvalues[ben_idx] < 0.05:
            compare_result.append(twof(svm_rank[ben_idx])+'*')
        else:
            compare_result.append(twof(svm_rank[ben_idx]))

    compare_result.append(twof(svm_rank[len(benchmarks)]))

    for ben_idx, ben in enumerate(benchmarks):
        if knn_pvalues[ben_idx] < 0.05:
            compare_result.append(twof(knn_rank[ben_idx])+'*')
        else:
            compare_result.append(twof(knn_rank[ben_idx]))

    compare_result.append(twof(knn_rank[len(benchmarks)]))

    friedman_results.append(compare_result)

header = [['Dataset', '', '', '', 'SVM', '', '', '', 'KNN'], ['','RFS', 'GFS', 'LOEC', 'DEEFS',
                                                              'RFS', 'GFS', 'LOEC', 'DEEFS']]
data = header+friedman_results
tabular = tables.Tabular(data)
table = tables.Table(tabular)
table.set_caption('Friedmen test: Comparing with embedded.')
table.set_label('tb:friedEmbed')
print(table)

# # perform friedman test JADE filter
# benchmarks = ['J-cor']
# master = ['H_l1_100']
# friedman_results = []
# for dataset_idx, dataset in enumerate(datasets):
#     knn_result = []
#     svm_result = []
#     for method in benchmarks+master:
#         knn_result.append(1.0-np.array(knn_accs[method][dataset_idx]))
#         svm_result.append(1.0-np.array(svm_accs[method][dataset_idx]))
#     _, _, knn_rank, knn_pivots = nontest.friedman_test(*knn_result)
#     _, _, svm_rank, svm_pivots = nontest.friedman_test(*svm_result)
#     knn_pivots_dict = {key: knn_pivots[i] for i, key in enumerate(benchmarks+master)}
#     _, _, _, knn_pvalues = nontest.holm_test(knn_pivots_dict)
#     svm_pivots_dict = {key: svm_pivots[i] for i, key in enumerate(benchmarks + master)}
#     _, _, _, svm_pvalues = nontest.holm_test(svm_pivots_dict)
#
#     compare_result = [dataset]
#     for ben_idx, ben in enumerate(benchmarks):
#         if svm_pvalues[ben_idx] < 0.05:
#             compare_result.append(twof(svm_rank[ben_idx])+'*')
#         else:
#             compare_result.append(twof(svm_rank[ben_idx]))
#
#     compare_result.append(twof(svm_rank[len(benchmarks)]))
#
#     for ben_idx, ben in enumerate(benchmarks):
#         if knn_pvalues[ben_idx] < 0.05:
#             compare_result.append(twof(knn_rank[ben_idx])+'*')
#         else:
#             compare_result.append(twof(knn_rank[ben_idx]))
#
#     compare_result.append(twof(knn_rank[len(benchmarks)]))
#
#     friedman_results.append(compare_result)
#
# header = [['Dataset', '', 'SVM', '', 'KNN'], ['',  'J-cor', 'DEEFS', 'J-cor', 'DEEFS']]
# data = header+friedman_results
# tabular = tables.Tabular(data)
# table = tables.Table(tabular)
# table.set_caption('Friedmen test: Comparing with wrapper.')
# table.set_label('tb:friedWrapper')
# print(table)

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

    tmp_ave = []
    for arr_time in times[method]:
        tmp_ave.append(np.mean(find_g0(arr_time)))
    times[method] = tmp_ave


# # Print data table
# header = [['', '#Features', '#Instances', 'Class ratio']]
# data = []
# for data_idx, dataset in enumerate(datasets):
#     data.append([dataset, data_info['nf'][data_idx], data_info['ni'][data_idx],
#                 twof(data_info['clp'][data_idx])+':'+twof(data_info['cln'][data_idx])])
# data = header+data
# tabular = tables.Tabular(data)
# table = tables.Table(tabular)
# table.set_caption('Dataset.')
# table.set_label('tb:dataset')
# print(table)

# # Print comparison with using all features
# header = [['', 'SVM acc', '', 'KNN acc', '', 'NF', ''],
#           ['Dataset', 'Full', 'DEEFS', 'Full', 'DEEFS', 'Full', 'DEEFS']]
# data = []
# for data_idx, dataset in enumerate(datasets):
#     row_data = []
#     row_data.append(dataset)
#     row_data.append(twof(svm_accs['Full'][data_idx]*100)+' '+svm_accs_sig['Full'][data_idx])
#     row_data.append(twof(svm_accs['H_l1_100'][data_idx]*100))
#     row_data.append(twof(knn_accs['Full'][data_idx]*100)+' '+knn_accs_sig['Full'][data_idx])
#     row_data.append(twof(knn_accs['H_l1_100'][data_idx]*100))
#     row_data.append(twof(nfs['Full'][data_idx]))
#     row_data.append(twof(nfs['H_l1_100'][data_idx]))
#     data.append(row_data)
# data = header+data
# tabular = tables.Tabular(data)
# table = tables.Table(tabular)
# table.set_caption('Compare with using all features.')
# table.set_label('tb:vsAll')
# print(table)

# # Print comparison with using wrapper
# header = [['', '', '', '','SVM acc', '', '', '', 'KNN acc', '', '', '', 'NF'],
#           ['Dataset', 'w90', 'w98', 'w100', 'DEEFS', 'w90', 'w98', 'w100', 'DEEFS', 'w90', 'w98', 'w100', 'DEEFS',]]
# data = []
# for data_idx, dataset in enumerate(datasets):
#     row_data = []
#     row_data.append(dataset)
#     row_data.append(twof(svm_accs['w90'][data_idx]*100)+' '+svm_accs_sig['w90'][data_idx])
#     row_data.append(twof(svm_accs['w98'][data_idx]*100)+' '+svm_accs_sig['w98'][data_idx])
#     row_data.append(twof(svm_accs['w100'][data_idx]*100)+' '+svm_accs_sig['w100'][data_idx])
#     row_data.append(twof(svm_accs['H_l1_100'][data_idx]*100))
#     row_data.append(twof(knn_accs['w90'][data_idx]*100)+' '+knn_accs_sig['w90'][data_idx])
#     row_data.append(twof(knn_accs['w98'][data_idx]*100)+' '+knn_accs_sig['w98'][data_idx])
#     row_data.append(twof(knn_accs['w100'][data_idx]*100)+' '+knn_accs_sig['w100'][data_idx])
#     row_data.append(twof(knn_accs['H_l1_100'][data_idx]*100))
#     row_data.append(twof(nfs['w90'][data_idx]))
#     row_data.append(twof(nfs['w98'][data_idx]))
#     row_data.append(twof(nfs['w100'][data_idx]))
#     row_data.append(twof(nfs['H_l1_100'][data_idx]))
#     data.append(row_data)
# data = header+data
# tabular = tables.Tabular(data)
# table = tables.Table(tabular)
# table.set_caption('Compare with wrapper.')
# table.set_label('tb:vsWrapper')
# print(table)

# Print comparison with using embedded
header = [['', '', '', '','SVM acc', '', '', '', 'KNN acc'],
          ['Dataset', 'RFS', 'GFS', 'LOEC', 'DEEFS', 'RFS', 'GFS', 'LOEC', 'DEEFS']]
data = []
for data_idx, dataset in enumerate(datasets):
    row_data = []
    row_data.append(dataset)
    row_data.append(twof(svm_accs['RFS'][data_idx]*100)+' '+svm_accs_sig['RFS'][data_idx])
    row_data.append(twof(svm_accs['GFS'][data_idx]*100)+' '+svm_accs_sig['GFS'][data_idx])
    row_data.append(twof(svm_accs['LOEC'][data_idx]*100)+' '+svm_accs_sig['LOEC'][data_idx])
    row_data.append(twof(svm_accs['H_l1_100'][data_idx]*100))
    row_data.append(twof(knn_accs['RFS'][data_idx]*100)+' '+knn_accs_sig['RFS'][data_idx])
    row_data.append(twof(knn_accs['GFS'][data_idx]*100)+' '+knn_accs_sig['GFS'][data_idx])
    row_data.append(twof(knn_accs['LOEC'][data_idx]*100)+' '+knn_accs_sig['LOEC'][data_idx])
    row_data.append(twof(knn_accs['H_l1_100'][data_idx]*100))
    data.append(row_data)
data = header+data
tabular = tables.Tabular(data)
table = tables.Table(tabular)
table.set_caption('Compare with embedded.')
table.set_label('tb:vsEmbedded')
print(table)

# Print comparison with using filter
header = [['', '', '', '','SVM acc', '', '', '', 'KNN acc'],
          ['Dataset', 'CFS', 'mRMR', 'reliefF', 'DEEFS', 'CFS', 'mRMR', 'reliefF', 'DEEFS']]
data = []
for data_idx, dataset in enumerate(datasets):
    row_data = []
    row_data.append(dataset)
    row_data.append(twof(svm_accs['CFS'][data_idx]*100)+' '+svm_accs_sig['CFS'][data_idx])
    row_data.append(twof(svm_accs['mRMR'][data_idx]*100)+' '+svm_accs_sig['mRMR'][data_idx])
    row_data.append(twof(svm_accs['reliefF'][data_idx]*100)+' '+svm_accs_sig['reliefF'][data_idx])
    row_data.append(twof(svm_accs['H_l1_100'][data_idx]*100))
    row_data.append(twof(knn_accs['CFS'][data_idx]*100)+' '+knn_accs_sig['CFS'][data_idx])
    row_data.append(twof(knn_accs['mRMR'][data_idx]*100)+' '+knn_accs_sig['mRMR'][data_idx])
    row_data.append(twof(knn_accs['reliefF'][data_idx]*100)+' '+knn_accs_sig['reliefF'][data_idx])
    row_data.append(twof(knn_accs['H_l1_100'][data_idx]*100))
    data.append(row_data)
data = header+data
tabular = tables.Tabular(data)
table = tables.Table(tabular)
table.set_caption('Compare with filter.')
table.set_label('tb:vsFilters')
print(table)


# Computation times
header = [['Dataset', 'CFS', 'mRMR', 'reliefF', 'RFS', 'GFS', 'LOEC', 'w90', 'DEEFS']]
data = []
for data_idx, dataset in enumerate(datasets):
    row_data = []
    row_data.append(dataset)
    row_data.append(twof(times['CFS'][data_idx]))
    row_data.append(twof(times['mRMR'][data_idx]))
    row_data.append(twof(times['reliefF'][data_idx]))
    row_data.append(twof(times['RFS'][data_idx]))
    row_data.append(twof(times['GFS'][data_idx]))
    row_data.append(twof(times['LOEC'][data_idx]))
    row_data.append(twof(times['w90'][data_idx]))
    row_data.append(twof(times['H_l1_100'][data_idx]))
    data.append(row_data)
data = header+data
tabular = tables.Tabular(data)
table = tables.Table(tabular)
table.set_caption('Computation time.')
table.set_label('tb:vsTime')
print(table)

# # print Jade filter
# header = [['', '','SVM acc', '', 'KNN acc',   '', 'NF'],
#           ['Dataset',  'J-cor', 'DEEFS', 'J-cor', 'DEEFS',  'J-cor', 'DEEFS',]]
# data = []
# for data_idx, dataset in enumerate(datasets):
#     row_data = []
#     row_data.append(dataset)
#     # row_data.append(twof(svm_accs['J-rel'][data_idx]*100)+' '+svm_accs_sig['J-rel'][data_idx])
#     row_data.append(twof(svm_accs['J-cor'][data_idx]*100)+' '+svm_accs_sig['J-cor'][data_idx])
#     row_data.append(twof(svm_accs['H_l1_100'][data_idx]*100))
#     # row_data.append(twof(knn_accs['J-rel'][data_idx]*100)+' '+knn_accs_sig['J-rel'][data_idx])
#     row_data.append(twof(knn_accs['J-cor'][data_idx]*100)+' '+knn_accs_sig['J-cor'][data_idx])
#     row_data.append(twof(knn_accs['H_l1_100'][data_idx]*100))
#     # row_data.append(twof(nfs['J-rel'][data_idx]))
#     row_data.append(twof(nfs['J-cor'][data_idx]))
#     row_data.append(twof(nfs['H_l1_100'][data_idx]))
#     data.append(row_data)
# data = header+data
# tabular = tables.Tabular(data)
# table = tables.Table(tabular)
# table.set_caption('Compare with filter using JADE.')
# table.set_label('tb:vsFilterJADE')
# print(table)


# # Computation times
# header = [['Dataset', 'J-cor', 'DEEFS']]
# data = []
# for data_idx, dataset in enumerate(datasets):
#     row_data = []
#     row_data.append(dataset)
#     # row_data.append(twof(times['J-rel'][data_idx]))
#     row_data.append(twof(times['J-cor'][data_idx]))
#     row_data.append(twof(times['H_l1_100'][data_idx]))
#     data.append(row_data)
# data = header+data
# tabular = tables.Tabular(data)
# table = tables.Table(tabular)
# table.set_caption('Computation time.')
# table.set_label('tb:vsTime')
# print(table)