import copy
import numpy as np
from tabulate import tabulate
from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    split_index = 0
    if random_state is not None:
        np.random.seed(random_state)
    if shuffle:
        myutils.randomize_in_place(X, y)
    if test_size >= 1:
        split_index = -test_size
    else:
        split_index = int(len(y) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []
    n_samples = len(X)
    indices = np.arange(0, n_samples, 1)
    if random_state is not None:
        np.random.seed(random_state)
    if shuffle:
        myutils.randomize_in_place(indices)
    num_larger_samples = n_samples % n_splits
    larger_split = n_samples // n_splits + 1
    curr_index = 0
    for i in range(n_splits):
        X_train_folds.append(list(indices[:curr_index]))
        if i < num_larger_samples:
            X_test_folds.append(list(indices[curr_index:curr_index + larger_split]))
            curr_index += larger_split
        else:
            X_test_folds.append(list(indices[curr_index:(curr_index + larger_split - 1)]))
            curr_index += larger_split - 1
        if curr_index < len(X):
            if len(X_train_folds[-1]) == 0:
                X_train_folds[-1] = list(indices[curr_index:])
            else:
                X_train_folds[-1] = X_train_folds[-1] + list(indices[curr_index:])
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    indices = list(np.arange(0, len(X)))
    if random_state is not None:
        np.random.seed(random_state)
    if shuffle:
        myutils.randomize_in_place(X, y)
    _, grouped_indices = myutils.group_by(indices, y)
    X_train_folds = [[] for _ in range(n_splits)]
    X_test_folds = [[] for _ in range(n_splits)]
    num_distributed = 0
    for group in grouped_indices:
        for index in group:
            for i in range(n_splits):
                if i != num_distributed % n_splits:
                    X_train_folds[i].append(int(index))
                else:
                    X_test_folds[i].append(int(index))
            num_distributed += 1
    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    if random_state is not None:
        np.random.seed(random_state)
    if n_samples is None:
        n_samples = len(X)
    X_sample = []
    X_out_of_bag = copy.deepcopy(X)
    if y is not None:
        y_sample = []
        y_out_of_bag = copy.deepcopy(y)
    else:
        y_sample = None
        y_out_of_bag = None
    sampled_indices = []
    for _ in range(n_samples):
        rand_index = np.random.randint(0, len(X))
        X_sample.append(X[rand_index])
        sampled_indices.append(rand_index)
        if y is not None:
            y_sample.append(y[rand_index])
    for index in list(sorted(set(sampled_indices), reverse=True)):
        X_out_of_bag.pop(index)
        if y is not None:
            y_out_of_bag.pop(index)
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0 for _ in labels] for _ in labels]
    for i in range(len(y_true)):
        matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    score = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            score += 1
    if normalize:
        score = score / len(y_true)
    return score

def find_eval_nums_for_k_fold(classifier, k, X, y, stratified=False):
    """Computes the accuracy of all the classifiers, using training and testing
        sets chosen using k-fold or stratified k-fold cross validation.

    Args:
        classifier: the classifier to fit/predict
        k (int): the number of folds.
        X (list of list of objects): the X data.
        y (list of objects): the y data (parallel to X).
        stratified (bool): whether to use stratified k-fold cross validation or not.

    Returns:
        accuracy (float), precision (float), recall (float), f1 (float)
    """
    accuracy = 0
    if stratified:
        X_train_folds, X_test_folds = stratified_kfold_cross_validation(X, y, k, 0, True)
    else:
        X_train_folds, X_test_folds = kfold_cross_validation(X, k, 0, True)
    entire_y_test = []
    entire_y_pred = []
    for i in range(k):
        X_train = [X[j] for j in X_train_folds[i]]
        y_train = [y[j] for j in X_train_folds[i]]
        X_test = [X[j] for j in X_test_folds[i]]
        y_test = [y[j] for j in X_test_folds[i]]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        entire_y_pred = entire_y_pred + y_pred
        entire_y_test = entire_y_test + y_test
        accuracy += accuracy_score(y_test, y_pred, False)
    precision = binary_precision_score(entire_y_test, entire_y_pred)
    recall = binary_recall_score(entire_y_test, entire_y_pred)
    f1 = binary_f1_score(entire_y_test, entire_y_pred)
    return accuracy / len(X), precision, recall, f1

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    tp = 0
    positives = 0
    if pos_label is None:
        if labels is not None:
            pos_label = labels[0]
        else:
            pos_label = y_true[0]
    for i in range(len(y_true)):
        if y_pred[i] == pos_label:
            positives += 1
            if y_true[i] == pos_label:
                tp += 1
    if positives == 0:
        return 0
    return tp / positives

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    tp = 0
    fn = 0
    if pos_label is None:
        if labels is not None:
            pos_label = labels[0]
        else:
            pos_label = y_true[0]
    for i in range(len(y_true)):
        if y_pred[i] == pos_label == y_true[i]:
            tp += 1
        elif y_pred[i] != pos_label and y_true[i] == pos_label:
            fn += 1
    if tp == fn == 0:
        return 0
    return tp / (tp + fn)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision == recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Added for the classification
def error_rate(y_true, y_pred, normalize=True):
    """Compute the classification prediction error rate.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    """
    number_failed = 0
    error = 0
    for i, item in enumerate(y_true):
        if item != y_pred[i]:
            number_failed += 1
    if normalize is True:
        error = number_failed / len(y_true)
    else:
        error = number_failed
    return error

def create_matrix_table(actual_values, predicted_values, matrix_label, label):
    """Creates a confusion matrix and uses tabulate to make it pretty
        Args:
            actual_values (list of obj): the true values of the dataset
            predicted_values (list of obj): the predicted values from the classifier
            matrix_label (list of obj): the unique items in the dataset
            label (list of obj): the label that goes at the top of the table
        Returns:
            table (tabulated table): the confusion matrix in table form
    """
    new_table = confusion_matrix(actual_values, predicted_values, matrix_label)

    for i, row in enumerate(new_table):
        totally_correct = 0
        total_number = 0
        for j, value in enumerate(row):
            total_number += value
            if i == j:
                totally_correct = value
        row.insert(0,matrix_label[i])
        row.append(total_number)
        try:
            row.append((totally_correct/total_number)*100)
        except:
            row.append(0)
    table = tabulate(new_table,headers=label)
    return table
