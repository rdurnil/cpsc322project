import numpy as np
from tabulate import tabulate
import myutils

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
    if labels is None:
        labels = myutils.find_unique_items(y_true)
    if pos_label is None:
        pos_label = labels[0]
    true_positive_value = 0
    false_positive_value = 0

    for i, item in enumerate(y_pred):
        if item == pos_label:
            if item == y_true[i]:
                true_positive_value += 1
            else:
                false_positive_value += 1
    if (true_positive_value + false_positive_value) == 0:
        precision = 0.0
    else:
        precision = true_positive_value / (true_positive_value + false_positive_value)
    return precision

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
    if labels is None:
        labels = myutils.find_unique_items(y_true)
    if pos_label is None:
        pos_label = labels[0]
    true_positive_value = 0
    false_negative_value = 0
    for i, item in enumerate(y_true):
        if item == pos_label:
            if item == y_pred[i]:
                true_positive_value += 1
            else:
                false_negative_value += 1
    if (true_positive_value + false_negative_value) == 0:
        recall = 0.0
    else:
        recall = true_positive_value / (true_positive_value + false_negative_value)
    return recall

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
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# from PA5 with the analysis removed

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
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle is True:
        myutils.randomize_in_place(X, y)

    n = len(X)
    if isinstance(test_size, float):
        split_index = int((1 - test_size) * n)
    elif isinstance(test_size, int):
        split_index = len(X) - test_size
    X_train , X_test = X[0:split_index], X[split_index:]
    y_train , y_test = y[0:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test


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
    if random_state is not None:
        np.random.seed(random_state)
    X_test_folds = []
    X_train_folds = []
    index_list = []
    for i, _ in enumerate(X):
        index_list.append(i)
    if shuffle is True:
        myutils.randomize_in_place(index_list, parallel_list=None)
    for i in range(n_splits):
        X_test_folds.append([])
        X_train_folds.append([])
    index_test = 0
    while index_test < len(index_list):
        for i, fold in enumerate(X_test_folds):
            if index_test >= len(index_list):
                break
            fold.append(index_list[index_test])
            index_test += 1
    for i, fold in enumerate(X_test_folds):
        for _, index in enumerate(index_list):
            if not myutils.determine_inclusion(fold, index):
                X_train_folds[i].append(index)
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
    if random_state is not None:
        np.random.seed(random_state)
    X_test_folds = []
    X_train_folds = []
    index_list = []
    X_indexes = []

    for i, item in enumerate(y):
        X_indexes.append([i, item])

    if shuffle is True:
        myutils.randomize_in_place(X_indexes)

    unique_lists = myutils.group_items(y, X_indexes)

    for i, _ in enumerate(X):
        index_list.append(i)

    for i in range(n_splits):
        X_test_folds.append([])
        X_train_folds.append([])

    for _, groups in enumerate(unique_lists):
        num_items_used = 0
        while num_items_used < len(groups):
            for _, folds in enumerate(X_test_folds):
                if num_items_used >= len(groups):
                    break
                folds.append(groups[num_items_used][0])
                num_items_used += 1

    for i, folds in enumerate(X_test_folds):
        temp_row = []
        for _, indexes in enumerate(index_list):
            if not myutils.determine_inclusion(folds, indexes):
                temp_row.append(indexes)
        if shuffle:
            myutils.randomize_in_place(temp_row)
        X_train_folds[i] = temp_row

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
    X_sample = []
    X_out_of_bag = []
    index_list = []
    if y is not None:
        y_sample = []
        y_out_of_bag = []
    else:
        y_sample = None
        y_out_of_bag = None

    if n_samples is None:
        n_samples = len(X)
    for i, _ in enumerate(X):
        index_list.append(i)

    for i in range(n_samples):
        choosen_index = np.random.randint(len(X))
        X_sample.append(X[choosen_index])
        if y is not None:
            y_sample.append(y[choosen_index])
    for i, item in enumerate(X):
        if not myutils.determine_inclusion(X_sample, item):
            X_out_of_bag.append(item)
            if y is not None:
                y_out_of_bag.append(y[i])

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
    matrix = []
    for _, _ in enumerate(labels):
        matrix.append([])
    for _, row in enumerate(matrix):
        for _, _ in enumerate(labels):
            row.append(0)

    for k, values in enumerate(y_true):
        matrix[labels.index(values)][labels.index(y_pred[k])] += 1

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
    number_predicted = 0
    accuracy = 0
    for i, item in enumerate(y_true):
        if item == y_pred[i]:
            number_predicted += 1
    if normalize is True:
        accuracy = number_predicted / len(y_true)
    else:
        accuracy = number_predicted
    return accuracy

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
