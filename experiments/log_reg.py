"""Logistic regression on modified dataset.
"""

import data
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


plt.style.use('seaborn')


def main():
    X_train, y_train, X_test, y_test = data.load('subsample', n=1000, pct_pos=0.75)
    print('Data subsampled.')
    clf   = LogisticRegression()
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)
    preds = preds[:, 1]  # Select the predictions for positive class.
    save_roc_curve_data(y_test, preds)
    acc   = metrics.accuracy_score(y_test, preds)
    prec  = metrics.precision_score(y_test, preds)
    rec   = metrics.recall_score(y_test, preds)
    f1    = metrics.f1_score(y_test, preds)
    auc   = metrics.roc_auc_score(y_test, preds)
    print(acc, prec, rec, f1, auc)
    with open('experiments/log_reg_results.csv', 'w+') as f:
        line = '%s,%s,%s,%s,%s' % (acc, prec, rec, f1, auc)
        f.write(line)


def save_roc_curve_data(y_test, preds):
    fpr, tpr, thresh = metrics.roc_curve(y_test, preds)
    plt.plot(fpr, tpr)
    plt.show()


if __name__ == '__main__':
    main()
