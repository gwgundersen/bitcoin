"""Support vector machine on modified dataset.
"""

import data
from sklearn import metrics, svm


def main():
    X_train, y_train, X_test, y_test = data.load('subsample', n=100000, pct_pos=0.7)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc  = metrics.accuracy_score(y_test, preds)
    prec = metrics.precision_score(y_test, preds)
    rec  = metrics.recall_score(y_test, preds)
    print(acc, prec, rec)


if __name__ == '__main__':
    main()
