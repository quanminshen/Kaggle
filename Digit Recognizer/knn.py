import numpy as np
import operator
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV


def load_train_data():
    data = np.loadtxt(open('data/train.csv'), dtype=np.float, delimiter=',', skiprows=20000)
    label = data[:, 0]
    data = data[:, 1:]
    return data, label


def load_test_data():
    data = np.loadtxt(open('data/test.csv'), dtype=np.float, delimiter=',', skiprows=1)
    return data


def classify(in_x, data_set, labels, k):
    in_x = np.mat(in_x)
    data_set = np.mat(data_set)
    labels = np.array(labels)
    data_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_size, 1)) - data_set
    square_diff_mat = np.array(diff_mat) ** 2
    square_dist = square_diff_mat.sum(axis=1)
    sorted_idx = square_dist.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[0, sorted_idx[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def save_results(result):
    idx = np.array([range(1, result.shape[0]+1)]).transpose()
    result = np.array([result]).transpose()
    np.savetxt('data/result.csv', np.hstack((idx, result)), fmt='%d', delimiter=',',
               header='ImageId,Label', comments='')


def knn_classify():
    train_data, train_label = load_train_data()
    test_data = load_test_data()
    k = np.linspace(1, 10, 10, dtype=np.int)
    parameters = {'n_neighbors': k, 'weights': ['uniform', 'distance']}
    knn = neighbors.KNeighborsClassifier()
    clf = GridSearchCV(estimator=knn, param_grid=parameters, cv=5, n_jobs=2)
    clf.fit(train_data, train_label)
    print('best score: ', clf.best_score_)
    print('best n_neighbors selection: ', clf.best_estimator_.n_neighbors)
    print('best weights selection: ', clf.best_estimator_.weights)
    # knn = neighbors.KNeighborsClassifier(neighbors=clf.best_estimator_.k)
    # knn.fit(train_data, train_label)
    # knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    # knn.fit(train_data, train_label)
    result = clf.predict(test_data)
    save_results(result)
    print('done!')

if __name__ == '__main__':
    knn_classify()




