#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import data_io


def svm_classify():
    t0 = time()
    train_data, train_label = data_io.load_train_data()
    test_data = data_io.load_test_data()
    print('load data in %0.3fs' % (time() - t0))
    print('Fitting the classifier to the training set')
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf.fit(train_data, train_label)
    print('done in %0.3fs' % (time() - t0))
    print('Best estimator found by grid search:')
    print(clf.best_estimator_)

    result = clf.predict(test_data)
    data_io.save_results(result)
    print('done!')

if __name__ == '__main__':
    svm_classify()





