"""
Authors: Bach Nguyen
Created: 07/06/2021
Description:
-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
"""
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import Base
from scipy.io import savemat

np.random.seed(1617)
d1 = 15
d2 = 70
W1 = np.array([0.9, -0.9, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1,
      0.1, -0.1, 0.1, -0.1, 0.05])
W2 = np.zeros(d1, dtype=float)
W3 = np.zeros(d2, dtype=float)
W = np.concatenate((W1, W2, W3), axis=0)
n = 600
X1 = np.random.rand(n, d1)
X2 = X1 + np.random.rand(n, d1)/10.0
X3 = np.random.rand(n, d2)
X = np.concatenate((X1, X2, X3), axis=1)
Y = np.sum(np.multiply(X, W), axis=1)
Y = Y > 0
Y_record = np.reshape(Y, (Y.shape[0], 1))
data = {'X': X, 'Y': Y_record}
savemat('/vol/grid-solar/sgeusers/nguyenhoai2/Dataset/FSMatlab/SyntheticPy.mat', data)

# ns = [100, 200, 300, 400, 500, 600]
# for n in ns:
#     print('----------n=%d----------' % n)
#     X1 = np.random.rand(n, d1)
#     X2 = X1 + np.random.rand(n, d1)/10.0
#     X3 = np.random.rand(n, d2)
#     X = np.concatenate((X1, X2, X3), axis=1)
#     Y = np.sum(np.multiply(X, W), axis=1)
#     Y = Y > 0
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1617, shuffle=True)
X_train, X_test = Base.normalise_data(X_train, X_test)
cs = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000, 10000]
citers = [1000, 10000, 100000]
for c in cs:
  for iter in citers:
      clf = svm.LinearSVC(random_state=1617, C=c, penalty='l2', max_iter=iter)
      clf.fit(X_train, y_train)
      svm_full_acc = balanced_accuracy_score(y_test, clf.predict(X_test))
      print('%f-%d:%f' % (c, iter, svm_full_acc))

