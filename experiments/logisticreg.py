"""Logistic regression on modified dataset.
"""

import data
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def main():
    X_train, y_train, X_test, y_test = data.load('subsample', n=10000000, pct_pos=0.01)
    clf   = LogisticRegression()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc   = metrics.accuracy_score(y_test, preds)
    prec  = metrics.precision_score(y_test, preds)
    rec   = metrics.recall_score(y_test, preds)
    f1    = metrics.f1_score(y_test, preds)
    auc   = metrics.roc_auc_score(y_test, preds)
    print(acc, prec, rec, f1, auc)
    with open('experiments/logisticreg_results.csv', 'w+') as f:
        line = '%s,%s,%s,%s,%s' % (acc, prec, rec, f1, auc)
        f.write(line)


if __name__ == '__main__':
    main()
