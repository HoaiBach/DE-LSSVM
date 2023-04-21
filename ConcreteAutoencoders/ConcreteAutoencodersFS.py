from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold, train_test_split
import Paras
import Base

from keras.datasets import mnist


def decoder(x):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(2)(x)
    return x

def convertVectorMatrix(label):
    y_m = np.zeros((len(label), 2))
    indices_first = np.where(label == -1)[0]
    y_m[indices_first, 0] = 1
    indices_sec = np.where(label == 1)[0]
    y_m[indices_sec, 1] = 1
    return y_m


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    run = int(sys.argv[2])

    #load data
    mat = scipy.io.loadmat(Paras.data_dir + 'FSMatlab/'+dataset+'.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]

    # ensure that y label is either -1 or 1
    num_class, count = np.unique(y, return_counts=True)
    n_classes = np.unique(y).shape[0]
    assert(n_classes == 2)
    min_class = np.min(count)
    unique_classes = np.unique(y)
    y[y == unique_classes[0]] = -1
    y[y == unique_classes[1]] = 1
    y = np.int8(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1617, stratify=y)
    y_train = convertVectorMatrix(y_train)
    y_test = convertVectorMatrix(y_test)
    X_train, X_test = Base.normalise_data(X_train, X_test)
    no_features = X_train.shape[1]

    selector = ConcreteAutoencoderFeatureSelector(K=20, output_function=decoder, num_epochs=800)
    selector.fit(X_train, y_train)

    print('finish')
