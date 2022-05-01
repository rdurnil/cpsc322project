

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
        y_predicted = []
        predicted_values = []
        for i, classifier in enumerate(self.classifier_forest):
            predicted_values.append(classifier.predict(X_test))

        for i, _ in enumerate(X_test): # this should iterate through the rows
            all_results = []
            for j, results in enumerate(predicted_values): # this should iterate through was classifier at the row and add the value to a list
                all_results.append(predicted_values[j][i]) # this should append the value from the classifier and the row to the holder list
            
        
        pass #TODO: Fix this
