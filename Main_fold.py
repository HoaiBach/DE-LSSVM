import JADE_Embed, JADE_Wrapper
import Problem
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import svm, preprocessing
from sklearn.metrics import balanced_accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
import Paras
import time

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    run = int(sys.argv[2])
    alg_style = sys.argv[3]
    Paras.fit_normalized = sys.argv[4] == 'norm'
    if alg_style == 'embed':
        Paras.alpha = float(sys.argv[5])
    elif alg_style == 'wrapper':
        Paras.w_wrapper = float(sys.argv[5])/100.0

    seed = 1617*run
    np.random.seed(seed)

    to_print = 'Style: %s \n' % alg_style
    max_iterations = 50
    to_print += 'Maximum number of iterations: %d \n' % max_iterations
    pop_size = 300
    to_print += 'Population size: %d \n' % pop_size
    to_print += 'Alpha: %f \n' % Paras.alpha
    to_print += 'Threshold: %f \n' % Paras.threshold
    to_print += 'Wrapper weight: %f \n' % Paras.w_wrapper
    to_print += 'Normalized fitness: %s \n' % str(Paras.fit_normalized)
    to_print += '============================================\n'

    #load data
    mat = scipy.io.loadmat('/home/nguyenhoai2/Grid/data/FSMathlab/'+dataset+'.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]

    # testing
    # X = -1 + np.random.rand(10, 5)*2
    # W = np.array([-2, 0.01, 0.2, 0.01, 0])
    # y = np.ravel(np.dot(X, np.reshape(W, (5, 1))))
    # y[y > 0] = 1
    # y[y < 0] = 0

    # ensure that y label is either -1 or 1
    num_class, count = np.unique(y, return_counts=True)
    n_classes = np.unique(y).shape[0]
    assert(n_classes == 2)
    min_class = np.min(count)
    unique_classes = np.unique(y)
    y[y == unique_classes[0]] = -1
    y[y == unique_classes[1]] = 1
    y = np.int8(y)

    no_folds = min(5, min_class)
    sfold = StratifiedKFold(n_splits=no_folds, shuffle=True, random_state=1617)

    fold_idx = 1
    ave_full_knn = 0.0
    ave_full_svm = 0.0
    ave_sel_knn = 0.0
    ave_sel_svm = 0.0
    ave_nf = 0.0
    ave_time = 0.0

    for train_index, test_index in sfold.split(X, y):
        to_print += '*********** Fold %d ***********\n' % fold_idx
        fold_idx += 1
        knn_full_acc = 0.0
        svm_full_acc = 0.0
        knn_sel_acc = 0.0
        svm_sel_acc = 0.0
        exe_time = 0.0
        f_selected = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_min = np.min(X_train, axis=0)
        X_max = np.max(X_train, axis=0)
        non_dup = np.where(X_min != X_max)[0]
        X_min = X_min[non_dup]
        X_max = X_max[non_dup]
        X_train = X_train[:, non_dup]
        X_test = X_test[:, non_dup]
        X_train = 2*(X_train - X_min) / (X_max - X_min)-1
        X_test = 2*(X_test - X_min) / (X_max - X_min)-1
        no_features = X_train.shape[1]

        if alg_style == 'embed':
            start = time.time()
            prob = Problem.FS_LSSVM(X_train, y_train)
            min_pos = np.array([-1.0, ]*(no_features+1)+[0.0, ]*no_features)
            max_pos = np.array([1.0, ]*(no_features+1)+[1.0, ]*no_features)
            de = JADE_Embed.JADE(problem=prob, popsize=pop_size, dims=2 * no_features + 1, maxiters=max_iterations,
                                 c=0.1, p=0.05, minpos=min_pos, maxpos=max_pos)
            best_sol, best_fit, evo_process = de.evolve()
            exe_time = time.time()-start
            to_print += evo_process
            to_print += '============================================\n'

            weight = best_sol[0:no_features]
            b = best_sol[no_features:no_features + 1]
            mask = best_sol[no_features + 1:]
            f_selected = np.where(mask > Paras.threshold)[0]
            X_train_sel = X_train[:, f_selected]
            X_test_sel = X_test[:, f_selected]

            to_print += 'Best solution: '
            for val in best_sol:
                to_print += str(val) + ', '
            to_print += '\n'

            to_print += 'Selected features: '
            for f_idx in f_selected:
                to_print += str(f_idx) + ', '
            to_print += '\n'

            output = np.dot(X_test, np.reshape(weight, (no_features, 1))) + b
            output[output > 0] = 1
            output[output < 0] = -1
            output = np.ravel(output)
            built_acc = balanced_accuracy_score(y_test, output)

            knn = KNN(metric='euclidean')
            knn.fit(X_train, y_train)
            knn_full_acc = balanced_accuracy_score(y_test, knn.predict(X_test))
            knn_full_train_acc = balanced_accuracy_score(y_train, knn.predict(X_train))
            knn.fit(X_train_sel, y_train)
            knn_sel_acc = balanced_accuracy_score(y_test, knn.predict(X_test_sel))
            knn_sel_train_acc = balanced_accuracy_score(y_train, knn.predict(X_train_sel))
            to_print += 'Full train KNN: %f \n' % knn_full_train_acc
            to_print += 'Sel train KNN: %f \n' % knn_sel_train_acc
            to_print += 'Full KNN: %f \n' % knn_full_acc
            to_print += 'Sel KNN: %f \n' % knn_sel_acc

            clf = svm.LinearSVC(random_state=seed, C=1.0, penalty='l2')
            clf.fit(X_train, y_train)
            svm_full_acc = balanced_accuracy_score(y_test, clf.predict(X_test))
            svm_full_train_acc = balanced_accuracy_score(y_train, clf.predict(X_train))
            clf.fit(X_train_sel, y_train)
            svm_sel_acc = balanced_accuracy_score(y_test, clf.predict(X_test_sel))
            svm_sel_train_acc = balanced_accuracy_score(y_train, clf.predict(X_train_sel))
            to_print += 'Full train SVM: %f \n' % svm_full_train_acc
            to_print += 'Sel train SVM: %f \n' % svm_sel_train_acc
            to_print += 'Full SVM: %f \n' % svm_full_acc
            to_print += 'Sel SVM: %f \n' % svm_sel_acc
            to_print += 'Number of selected features: %d \n' % len(f_selected)
            to_print += 'Time: %f \n' % exe_time

            to_print += '=================================\n'
            to_print += 'Built SVM: %f \n' % built_acc
            clf = svm.LinearSVC(random_state=1617, C=1.0, penalty='l1', dual=False)
            clf.fit(X_train, y_train)
            svm_full_acc = balanced_accuracy_score(y_test, clf.predict(X_test))
            clf.fit(X_train_sel, y_train)
            svm_sel_acc = balanced_accuracy_score(y_test, clf.predict(X_test_sel))
            to_print += 'Full SVM l1: %f \n' % svm_full_acc
            to_print += 'Sel SVM l1: %f \n' % svm_sel_acc
            to_print += 'Number of selected features by SVM l1: %d \n' % len(f_selected)

        elif alg_style == 'wrapper':
            clf = svm.LinearSVC(random_state=seed, C=1.0, penalty='l2')
            start = time.time()
            prob = Problem.FS_Wrapper(X_train, y_train, clf)
            min_pos = np.array([0, ]*no_features)
            max_pos = np.array([1, ]*no_features)
            de = JADE_Wrapper.JADE(problem=prob, popsize=pop_size, dims=no_features, maxiters=max_iterations,
                                 c=0.1, p=0.05, minpos=min_pos, maxpos=max_pos)
            best_sol, best_fit, evo_process = de.evolve()
            exe_time = time.time() - start
            to_print += evo_process
            to_print += '============================================\n'

            f_selected = np.where(best_sol > Paras.threshold)[0]
            X_train_sel = X_train[:, f_selected]
            X_test_sel = X_test[:, f_selected]

            to_print += 'Best solution: '
            for val in best_sol:
                to_print += str(val) + ', '
            to_print += '\n'
            to_print += 'Selected features: '
            for f_idx in f_selected:
                to_print += str(f_idx) + ', '
            to_print += '\n'

            knn = KNN(metric='euclidean')
            knn.fit(X_train, y_train)
            knn_full_acc = balanced_accuracy_score(y_test, knn.predict(X_test))
            knn_full_train_acc = balanced_accuracy_score(y_train, knn.predict(X_train))
            knn.fit(X_train_sel, y_train)
            knn_sel_acc = balanced_accuracy_score(y_test, knn.predict(X_test_sel))
            knn_sel_train_acc = balanced_accuracy_score(y_train, knn.predict(X_train_sel))
            to_print += 'Full train KNN: %f \n' % knn_full_train_acc
            to_print += 'Sel train KNN: %f \n' % knn_sel_train_acc
            to_print += 'Full KNN: %f \n' % knn_full_acc
            to_print += 'Sel KNN: %f \n' % knn_sel_acc

            clf = svm.LinearSVC(random_state=seed, C=1.0, penalty='l2')
            clf.fit(X_train, y_train)
            svm_full_acc = balanced_accuracy_score(y_test, clf.predict(X_test))
            svm_full_train_acc = balanced_accuracy_score(y_train, clf.predict(X_train))
            clf.fit(X_train_sel, y_train)
            svm_sel_acc = balanced_accuracy_score(y_test, clf.predict(X_test_sel))
            svm_sel_train_acc = balanced_accuracy_score(y_train, clf.predict(X_train_sel))
            to_print += 'Full train SVM: %f \n' % svm_full_train_acc
            to_print += 'Sel train SVM: %f \n' % svm_sel_train_acc
            to_print += 'Full SVM: %f \n' % svm_full_acc
            to_print += 'Sel SVM: %f \n' % svm_sel_acc
            to_print += 'Number of selected features: %d \n' % len(f_selected)
            to_print += 'Time: %f \n' % exe_time

        ave_full_knn += knn_full_acc
        ave_full_svm += svm_full_acc
        ave_sel_knn += knn_sel_acc
        ave_sel_svm += svm_sel_acc
        ave_time += ave_time
        ave_nf += float(len(f_selected))

    ave_nf /= no_folds
    ave_sel_svm /= no_folds
    ave_sel_knn /= no_folds
    ave_full_svm /= no_folds
    ave_full_knn /= no_folds
    ave_time /= no_folds

    to_print += '***********************************Final results*****************************\n'
    to_print += 'Full SVM: %f \n' % ave_full_svm
    to_print += 'Sel SVM: %f \n' % ave_sel_svm
    to_print += 'Full KNN: %f \n' % ave_full_knn
    to_print += 'Sel KNN: %f \n' % ave_sel_knn
    to_print += 'Number of selected features: %f \n' % ave_nf

    f = open(str(run)+'.txt', 'w')
    f.write(to_print)
    f.close()
