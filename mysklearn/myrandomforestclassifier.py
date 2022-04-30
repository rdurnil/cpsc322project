

class MyRandomForestClassifier:
    """Represents a simple linear regressor.
    Attributes:
        n_values(int): the number of "weak" classifiers
        m_values(int): the number of "better" learners
        f_values(int): the number of random attribute subsets
    Notes:
        Loosely based on sklearn's LinearRegression:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
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

    def fit(self, X_train, y_train):
        """Creates a random forest of classifiers to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        pass #TODO: Fix this

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        pass #TODO: Fix this
