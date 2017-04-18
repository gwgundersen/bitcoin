"""Stochastic gradient descent classifier learned online.
"""

import data
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
import numpy as np


def run_experiments(n_batches):

    # Online algorithms in sklearn:
    # http://scikit-learn.org/stable/modules/scaling_strategies.html
    clf = SGDClassifier()
    with open('sgd_out.txt', 'w+') as f:
        for i in range(n_batches):
            acc, prec, rec, f1, auc = run_experiment(clf)
            print(acc, prec, rec, f1, auc)
            line = '%s,%s,%s,%s,%s' % (acc, prec, rec, f1, auc)
            f.write(line)
            if i == n_batches-1:
                print('FINAL -------')
                print(acc, prec, rec, f1, auc)

def run_experiment(clf):
    X_train, y_train, X_test, y_test = data.load('subsample', n=100000, pct_pos=0.01)
    X_train = np.delete(X_train, [0, 1], axis=1)
    X_test = np.delete(X_test, [0, 1], axis=1)
    print('Data subsampled.')
    clf.partial_fit(X_train, y_train, classes=[0, 1])
    preds = clf.predict(X_test)
    acc   = metrics.accuracy_score(y_test, preds)
    prec  = metrics.precision_score(y_test, preds)
    rec   = metrics.recall_score(y_test, preds)
    f1    = metrics.f1_score(y_test, preds)
    auc   = metrics.roc_auc_score(y_test, preds)
    # with open('experiments/logisticreg_results.csv', 'w+') as f:
    #     line = '%s,%s,%s,%s,%s' % (acc, prec, rec, f1, auc)
    #     f.write(line)
    return acc, prec, rec, f1, auc


if __name__ == '__main__':
    N_BATCHES = 100
    run_experiments(N_BATCHES)
