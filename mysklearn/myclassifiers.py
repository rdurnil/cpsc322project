import math
import operator
import numpy as np
from mysklearn import myutils
from mysklearn.mypytable import MyPyTable

 
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
        header(list): the create attribute names
        attribute_domains(dict): dictionary of unique values in the columns
        f_value(int): the number of attributes to calculate entropy for
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, forest=False, f_val = 2):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domains = None
        self.f_value = f_val
        self.forest_inclusion = forest

    def select_attribute(self, instances, attributes):
        """Selects the attribute to split on using entropy
            Args:
                instances (list of list): the rows of instances
                attributes (list of obj): the column that need entropy calculated
            Returns:
                The attribute that has the lowest entropy
        """
        entropy_options = []
        for _, att in enumerate(attributes):
            entropy_calc = self.calculate_entropy(instances, self.header.index(att))
            entropy_options.append(entropy_calc)
        att_index = myutils.find_smallest_value(entropy_options)
        return attributes[att_index]

    def calculate_entropy(self, instances, att_col):
        """Calculates the entropy of a single column
            Args:
                instances (list of list): the rows of instances
                att_col (int): the column to calculate the entropy of
            Returns:
                The weighted sum of the column
        """
        unique_class_vals = myutils.find_unique_items(self.y_train)
        num_class_options = len(unique_class_vals)
        entropy_total = 0
        for _, value in enumerate(self.attribute_domains[att_col]):
            total_class_options = myutils.create_filled_list(num_class_options, 0)
            for _, row in enumerate(instances):
                if row[att_col] == value:
                    total_class_options[unique_class_vals.index(row[-1])] += 1
            entropy_calc = 0
            for _, total in enumerate(total_class_options):
                if total == 0:
                    entropy_calc += 0
                else:
                    proportion = total / sum(total_class_options)
                    entropy_calc += -proportion * math.log2(proportion)
            entropy_total += (sum(total_class_options) / len(self.y_train)) * entropy_calc
        return entropy_total


    def partition_instances(self, instances, split_attribute):
        """Groups the instances by class value and puts them in a dictionary
            Args:
                instances (list of list): the instance rows
                split_attribute (str): the header of the attrubute that is being split on
        """
        # let's use a dictionary
        partitions = {} # key (string): value (subtable)
        att_index = self.header.index(split_attribute) # e.g. 0 for level
        att_domain = self.attribute_domains[att_index] # e.g. ["Junior", "Mid", "Senior"]
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)
        return partitions

    def all_same_class(self, partition):
        """Checks the partition to see if all the classes in the partition are the same
            Args:
                partition (dict): the dictionary that holds the instances
        """
        same_class = True
        class_type = partition[0][-1]
        for i in range(1, len(partition)):
            if partition[i][-1] != class_type:
                same_class = False
        return same_class

    def vote_class(self, instances):
        """Iterates through a list of list to find the most common class
            Args:
                instances (list of list of obj): the list of instances that hold the class labels
            Returns:
                The most common class in the list.
        """
        class_values = []
        for _, values in enumerate(instances):
            class_values.append(values[-1])
        unique_classes = myutils.find_unique_items(class_values)
        num_class_vals = myutils.count_unique_items(class_values, unique_classes)
        return unique_classes[myutils.find_largest_value(num_class_vals)]

    def vote_dict_class(self, partitions):
        """Called from case two, iterates through the dictionary values to find the most
        common class value
            Args:
                partitions (dict): the partition dictionary
            Returns:
                the class that appears the most in the dictionary
        """
        class_values = []
        for key in partitions.keys():
            for _, lists in enumerate(partitions[key]):
                class_values.append(lists[-1])
        unique_classes = myutils.find_unique_items(class_values)
        num_class_vals = myutils.count_unique_items(class_values, unique_classes)
        return unique_classes[myutils.find_largest_value(num_class_vals)]

    def compute_random_subset(self, values):
        # there is a function np.random.choice()
        values_copy = values[:] # shallow copy
        subset = set()
        if self.f_value <= len(values_copy):
            return values_copy
        else:
            while(len(subset) < self.f_value):
                subset.append(values_copy[np.random.randint(0, len(values_copy))])
            return list(subset)

    def tdidt(self, current_instances, available_attributes, previous_length):
        """Recursively creates a decision tree
            Args:
                current_instances (list of list): the rows left to be classified
                avalable_attributes (list of str): the columns avalable to split on
                previous_length (int): the number of instances in the previous node in case
                    of a case three
        """
        if self.forest_inclusion is True:
            f_attributes = self.compute_random_subset(available_attributes)
        else: 
            f_attributes = available_attributes
        attribute = self.select_attribute(current_instances, f_attributes)
        available_attributes.remove(attribute)
        tree = ["Attribute", attribute]
        temp_tree = []
        case_three = False
        partitions = self.partition_instances(current_instances, attribute)
        for att_value, att_partion in partitions.items():
            value_subtree = ["Value", att_value]

            if len(att_partion) > 0 and self.all_same_class(att_partion):
                case_one_leaf = ["Leaf", att_partion[0][-1], len(att_partion), len(current_instances)]
                value_subtree.append(case_one_leaf)

            elif len(att_partion) > 0 and len(available_attributes) == 0:
                case_two_leaf = ["Leaf", self.vote_dict_class(partitions), len(att_partion), len(current_instances)]
                value_subtree.append(case_two_leaf)

            elif len(att_partion) == 0:
                case_three = True
                break

            else:
                subtree = self.tdidt(att_partion, available_attributes.copy(), len(current_instances))
                value_subtree.append(subtree)
            temp_tree.append(value_subtree)

        if case_three:
            tree = ["Leaf", self.vote_class(current_instances), len(current_instances), previous_length] # how do I get the second value
        else:
            tree = tree + temp_tree
        return tree

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.set_up_constants(X_train, y_train)
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = self.header.copy()
        self.tree = self.tdidt(train, available_attributes, len(train))

    def set_up_constants(self, x_vals, y_vals):
        """Creates the header and the attribute domains for the tdidt classifier plus save
        the class list
            Args:
                x_vals (list of list of obj): the x_values that are being classified
                y_vals (list of obj): the class values that are parallel to x_vals
        """
        self.header = []
        self.y_train = y_vals
        self.attribute_domains = {}
        for i in range(len(x_vals[0])):
            self.header.append("att" + str(i))
        for i in range(len(x_vals[0])):
            temp_col_list = []
            for _, row in enumerate(x_vals):
                temp_col_list.append(row[i])
            self.attribute_domains.update({i:myutils.find_unique_items(temp_col_list)})

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for _, item in enumerate(X_test):
            y_predicted.append(self.tdidt_predict(self.tree, item))
        return y_predicted

    def tdidt_predict(self, tree, instance):
        """Iterates through parts of the tree recursively untill it stumbles across a leaf node
            Args:
                tree (list of list): the part of the tree that is being searched
                instance (obj): the list of attributes that the prediction is for
            Returns:
                The class value that is being predicted.
        """
        info_type = tree[0]
        if info_type == "Leaf":
            return tree[1]
        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(value_list[2], instance)


    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rule = self.create_rule_string( self.tree, "IF", class_name, attribute_names)

    def create_rule_string(self, tree, rule_str, class_name, att_names):
        """Iterates through all possible tree paths appending the rules to a string which prints
        when a leaf node is reached
            Args:
                tree (list of list): the subtree that is currently being transversed
                rule_str (str): the running string that holds the rules in the tree path
                class_name (str): the name that the caller wants the class lable to be
                att_name (list of str): None if the caller wants the default att names (att0, att1...)
                    else, the names of the attribute names
        """
        info_type = tree[0]
        if info_type == "Leaf":
            print(rule_str + " THEN " + class_name + "=" + str(tree[1]))
            return 0
        for i in range(2, len(tree)):
            temp_string = ""
            value_list = tree[i]
            if att_names is not None:
                if rule_str == "IF":
                    temp_string = rule_str + " " + att_names[self.header.index(tree[1])] + "=" + value_list[1]
                else:
                    temp_string = rule_str + " AND " + att_names[self.header.index(tree[1])] + "=" + value_list[1]
            else:
                if rule_str == "IF":
                    temp_string = rule_str + " " + str(tree[1]) + "=" + value_list[1]
                else:
                    temp_string = rule_str + " AND " + str(tree[1]) + "=" + value_list[1]
            x = self.create_rule_string(value_list[2], temp_string, class_name, att_names)
        return x

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).
        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # todo (BONUS) fix this

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def calculate_priors(self, result_values):
        """Calculates the prior probabilities and returns them in a list along
        with a list of the unique prior options
            Args:
                result_values (list of object): the 1D list to find the priors in
            Returns:
                found_priors (dictionary): a dictionary containing a list of the unique values, a list of
                    the amount of the unique values, and a list of the priors values. All parallel lists
        """
        different_vals_y = myutils.find_unique_items(result_values)
        number_diff_y = myutils.count_unique_items(result_values, different_vals_y)
        self.priors = {}
        found_priors = []
        for i, _ in enumerate(different_vals_y):
            found_priors.append(number_diff_y[i] / len(result_values))
        self.priors.update({"unique items": different_vals_y})
        self.priors.update({"number items": number_diff_y})
        self.priors.update({"priors": found_priors})

    def calculate_posteriors(self, x_values, y_values):
        """Takes in the columns that need posteriors and calculates them
            Args:
                x_values (list of list of obj): the data that needs posteriors calculated
            Returns:
                posteriors (list of dict): a list of dictionariues that contains a list of the unique items,
                    a list of the number of unique items, and the posteriors calculated for the items.
        """
        header_list = []
        self.posteriors = []
        for i in range(len(x_values[0])):
            header_list.append(i)
        data_table = MyPyTable(header_list, x_values)

        for i,_ in enumerate(header_list):
            temp_dict = {}
            temp_list = data_table.get_column(i, False)
            temp_unique_items = myutils.find_unique_items(temp_list)
            temp_number_items = myutils.count_unique_items(temp_list, temp_unique_items)
            temp_post_list = []
            for _, item in enumerate(temp_unique_items):
                individual_post_list = [0] * len(self.priors["unique items"])
                for k, value in enumerate(temp_list):
                    if item is value:
                        result_value = y_values[k]
                        individual_post_list[self.priors["unique items"].index(result_value)] += 1
                for j, count in enumerate(individual_post_list):
                    individual_post_list[j] = (count / self.priors["number items"][j])
                temp_post_list.append(individual_post_list)
            temp_dict.update({"unique items": temp_unique_items})
            temp_dict.update({"number items": temp_number_items})
            temp_dict.update({"posteriors": temp_post_list})
            self.posteriors.append(temp_dict)

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.calculate_priors(y_train)
        self.calculate_posteriors(X_train, y_train)

    def calculate_probability(self, attribute_set):
        """Calculates all possible probabilities for each attribute in the given set
            Args:
                attribute_set (list of obj): a single test case to search the posteriors for probabilities
            Returns:
                num_probs (list of float): a list of probabilities for each unique prior
        """
        num_probs = [0] * len(self.priors["unique items"])

        for i, _ in enumerate(num_probs):
            current_probability = 1.0
            for j, item in enumerate(attribute_set):
                item_pos = self.posteriors[j]["unique items"].index(item)
                current_probability = current_probability * self.posteriors[j]["posteriors"][item_pos][i]
            current_probability = current_probability * self.priors["priors"][i]
            num_probs[i] = current_probability
        return num_probs

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for i, _ in enumerate(X_test):
            temp_value = self.calculate_probability(X_test[i])
            index_of_prob = myutils.find_largest_value(temp_value)
            y_predicted.append(self.priors["unique items"][index_of_prob])

        return y_predicted

# Previously Created Classifiers

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).
    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data
    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.
        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        if regressor is None:
            regressor = MySimpleLinearRegressor()
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predicted_y_values = self.regressor.predict(X_test)
        classified_y = []

        for i,_ in enumerate(predicted_y_values):
            classified_y.append(self.discretizer(predicted_y_values[i]))
        return classified_y

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train


    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        distances = []
        neighbor_indices = []
        for _, text_case in enumerate(X_test):
            test_total_dists = []
            row_indexes_dists = []
            temp_distances = []
            temp_neighbor = []
            for i, train_instance in enumerate(self.X_train):
                dist = self.compute_euclidean_distance(train_instance, text_case)
                row_indexes_dists.append([i, dist])
            row_indexes_dists.sort(key=operator.itemgetter(-1))
            test_total_dists.append(row_indexes_dists[:self.n_neighbors])
            for i, item in enumerate(test_total_dists):
                for k,_ in enumerate(item):
                    temp_distances.append(item[k][1])
                    temp_neighbor.append(item[k][0])
            distances.append(temp_distances)
            neighbor_indices.append(temp_neighbor)
        return distances, neighbor_indices

    def compute_euclidean_distance(self,v1, v2):
        """Computes the euclidian distance of two vectors
            Args:
                v1 (list of int): vector 1
                v2 (list of int): vector 2
            Returns:
                The euclidian distance between the vectors
        """
        dist = 0
        for i,_ in enumerate(v1):
            if isinstance(v1[i], str):
                if v1[i] == v2[i]:
                    dist += 1
                else:
                    dist += 0
            else:
                dist += ((v1[i] - v2[i]) ** 2)
        dist = np.sqrt(dist)
        return dist

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, indices = self.kneighbors(X_test)

        classification_strings = []
        temp_class_table = []

        for _, groups in enumerate(indices): # This goes through the index groups aka the individual test lists
            temp_class_row = []
            for _, ids in enumerate(groups): # transverses the indexes in the list
                temp_class_row.append(self.y_train[ids])
            temp_class_table.append(temp_class_row)

        for _, groups in enumerate(temp_class_table): # This goes through the index groups aka the individual test lists
            temp_unique = self.find_unique_items(groups)
            number_unique = self.count_unique_items(groups, temp_unique)
            classifier_found = temp_unique[number_unique.index(max(number_unique))]
            classification_strings.append(classifier_found)
        return classification_strings

    def count_unique_items(self, data_set, unique_items):
        """Counts the number of unique items in a 1D list
            Args:
                data_set (list): the list being searched
                unique_items (list): the items being searched for
            Returns:
            A list of the number of each unique item
        """
        count_list = []
        for i,_ in enumerate(unique_items):
            num_this_item = 0
            for j,_ in enumerate(data_set):
                if unique_items[i] == data_set[j]:
                    num_this_item += 1
            count_list.append(num_this_item)
        return count_list

    def find_unique_items(self, item_list):
        """Finds the unique items in a list
            Args:
                item_list (list of objs): list of objects to search
        """
        unique_list = []
        for _, item in enumerate(item_list):
            if item not in unique_list:
                unique_list.append(item)
        return unique_list

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        unique_items = []
        count_list = []
        for i, item in enumerate(y_train):
            if str(item) not in unique_items:
                unique_items.append(str(item))
        for i,_ in enumerate(unique_items):
            num_this_item = 0
            for j,_ in enumerate(y_train):
                if unique_items[i] == y_train[j]:
                    num_this_item += 1
            count_list.append(num_this_item)
        most_common_index = count_list.index(max(count_list))
        self.most_common_label = unique_items[most_common_index]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        dummy_classifiers = []

        for _,_ in enumerate(X_test):
            dummy_classifiers.append(self.most_common_label)

        return dummy_classifiers

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.
    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b
    Notes:
        Loosely based on sklearn's LinearRegression:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.
        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        X_train = [x[0] for x in X_train] # convert 2D list with 1 col to 1D list
        self.slope, self.intercept = MySimpleLinearRegressor.compute_slope_intercept(X_train,
            y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        predictions = []
        if self.slope is not None and self.intercept is not None:
            for test_instance in X_test:
                predictions.append(self.slope * test_instance[0] + self.intercept)
        return predictions

    @staticmethod # decorator to denote this is a static (class-level) method
    def compute_slope_intercept(x, y):
        """Fits a simple univariate line y = mx + b to the provided x y data.
        Follows the least squares approach for simple linear regression.
        Args:
            x(list of numeric vals): The list of x values
            y(list of numeric vals): The list of y values
        Returns:
            m(float): The slope of the line fit to x and y
            b(float): The intercept of the line fit to x and y
        """
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) \
            / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
        # y = mx + b => y - mx
        b = mean_y - m * mean_x
        return m, b