from mysklearn import myutils
from mysklearn import myevaluation
from mysklearn.myclassifiers import MyDecisionTreeClassifier
import numpy as np

class MyRandomForestClassifier:
    """Represents a simple linear regressor.
    Attributes:
        n_values(int): the number of "weak" classifiers
        m_values(int): the number of "better" learners
        f_values(int): the number of random attribute subsets
        classifier_forest(lst of decision trees): the decision trees that make up the forest
    """
    def __init__(self, n_val=100, m_val=10, f_val=2):
        """Initializer for MyRandomForestClassifier.

        Args:
            n_val(int): the number of "weak" classifiers
            m_val(int): the number of "better" learners
            f_val(int): the number of random attribute subsets
        """
        self.n_value = n_val
        self.m_value = m_val
        self.f_value = f_val
        self.classifier_forest = None 

    def fit(self, X_train, y_train):
        """Creates a random forest of tdidt classifiers to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        np.random.seed(0)
        trees = []
        tree_accuracy = []
        for n in range(self.n_value):
            build_indices, test_indices = myutils.compute_bootstrapped_sample(X_train)
            X_build, y_build = [], []
            for index in build_indices:
                X_build.append(X_train[index])
                y_build.append(y_train[index])
            tree = MyDecisionTreeClassifier()
            tree.fit(X_build, y_build)
            trees.append(tree)
            X_test, y_test = [], []
            for index in test_indices:
                X_test.append(X_train[index])
                y_test.append(y_train[index])
            y_pred = tree.predict(X_test)
            tree_accuracy.append(myevaluation.accuracy_score(y_test, y_pred, True))  
        myutils.sort_in_place(tree_accuracy, trees)
        self.classifier_forest = trees[-self.m_value:]

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        predicted_values = []
        for i, classifier in enumerate(self.classifier_forest):
            predicted_values.append(classifier.predict(X_test))

        for i, _ in enumerate(X_test): # this should iterate through the rows
            all_results = []
            for j, _ in enumerate(predicted_values): # this should iterate through was classifier at the row and add the value to a list
                all_results.append(predicted_values[j][i]) # this should append the value from the classifier and the row to the holder list
            y_predicted.append(myutils.vote_on_class(all_results)) # this takes the list of  class results and returns the voted class lable

        return y_predicted
