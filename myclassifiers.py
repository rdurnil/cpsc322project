"""
Programmer: Rie Durnil
Class: CPSC 322-01, Spring 2022
Programming Assignment #7
4/14/22

Description: This program implements four different classifiers:
kNN, Dummy, Naive Bayes, and Decision Trees.
"""

import operator
import math
import myutils as myutils

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

        for x_instance in X_test:
            index_distances = []
            for i, train_instance in enumerate(self.X_train):
                dist = myutils.compute_distance(train_instance, x_instance)
                index_distances.append([i, dist])
            index_distances.sort(key=operator.itemgetter(-1))
            top_n = index_distances[:self.n_neighbors]
            indices = []
            dists = []
            for i in range(self.n_neighbors):
                indices.append(top_n[i][0])
                dists.append(top_n[i][1])
            distances.append(dists)
            neighbor_indices.append(indices)
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indices = self.kneighbors(X_test)
        y_predicted = []
        for x_index in neighbor_indices:
            y_options = []
            counts = []
            for index in x_index:
                try:
                    option_index = y_options.index(self.y_train[index])
                    counts[option_index] += 1
                except ValueError:
                    y_options.append(self.y_train[index])
                    counts.append(1)
            y_predicted.append(y_options[counts.index(max(counts))])
        return y_predicted

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
        class_labels = []
        counts = []
        y_train.sort()
        for y in y_train:
            if len(class_labels) == 0:
                class_labels.append(y)
                counts.append(1)
            elif class_labels[-1] == y:
                counts[-1] += 1
            else:
                class_labels.append(y)
                counts.append(1)
        self.most_common_label = class_labels[counts.index(max(counts))]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for _ in X_test:
            y_predicted.append(self.most_common_label)
        return y_predicted

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
        posts = {}
        class_labels = {}
        for y in y_train:
            try:
                class_labels[y] = class_labels[y] + 1
            except KeyError:
                class_labels[y] = 1
        single_row_atts = {}
        for val in class_labels:
            single_row_atts[val] = 0
        for i in range(len(X_train[0])):
            posts[i] = {}
        for i, x in enumerate(X_train):
            for j in range(len(x)):
                try:
                    posts[j][x[j]][y_train[i]] = posts.get(j).get(x[j]).get(y_train[i]) + 1
                except AttributeError:
                    posts[j][x[j]] = single_row_atts.copy()
                    posts[j][x[j]][y_train[i]] = posts.get(j).get(x[j]).get(y_train[i]) + 1
        for a in posts:
            for b in posts[a]:
                for c in posts[a][b]:
                    posts[a][b][c] = posts[a][b][c] / class_labels[c]
        for c in class_labels:
            class_labels[c] = class_labels[c] / len(y_train)
        self.priors = class_labels
        self.posteriors = posts

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        probs = {}
        for class_label in self.priors:
            probs[class_label] = self.priors[class_label]
        for X in X_test:
            instance_probs = probs.copy()
            for i in range(len(X)):
                for inst in instance_probs:
                    instance_probs[inst] = instance_probs[inst] * self.posteriors[i][X[i]][inst]
            max_key = ""
            max_val = 0
            for inst in instance_probs:
                if instance_probs[inst] > max_val:
                    max_key = inst
                    max_val = instance_probs[inst]
                elif instance_probs[inst] == max_val:
                    if max_key > inst:
                        max_key = inst
                        max_val = instance_probs[inst]
            y_pred.append(max_key)
        return y_pred

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.att_domain = None

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
        self.X_train = X_train
        self.y_train = y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        train_copy = train.copy()
        header = []
        attribute_domains = {}

        for i in range(len(train_copy[0]) - 1):
            header.append("att" + str(i))
            attribute_domains[header[i]] = []
            for j in range(len(train_copy)):
                attribute_domains[header[i]].append(train_copy[j][i])
            attribute_domains[header[i]] = list(set(attribute_domains[header[i]]))
            attribute_domains[header[i]].sort()

        self.header = header
        self.att_domain = attribute_domains
        available_attributes = header.copy()
        self.tree = self.tdidt(train, available_attributes)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        for x in X_test:
            y_pred.append(self.tdidt_predict(self.tree, x))
        return y_pred

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
        all_path = []
        self.traverse_each_path(self.tree, [], all_path)
        for path in all_path:
            if attribute_names is None:
                for i in range(0, len(path) - 1, 2):
                    if i == 0:
                        print("IF", path[i], "==", path[i + 1], end=" ")
                    else:
                        print("AND", path[i], "==", path[i + 1], end=" ")
            else:
                for i in range(0, len(path) - 1, 2):
                    if i == 0:
                        print("IF", attribute_names[self.header.index(path[i])], "=", path[i + 1], end=" ")
                    else:
                        print("AND", attribute_names[self.header.index(path[i])], "=", path[i + 1], end=" ")
            print("THEN", class_name, "=", path[-1])

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
        pass # TODO: (BONUS) fix this

    def tdidt(self, current_instances, available_attributes, prev_len=None):
        """Implements the TDIDT algorithm to fit a decision tree.

        Args:
            current_instances(list of list of obj): The list of testing samples left.
            available_attributes(list of obj): The list of attributes that have not been
                split on.
            prev_len(int): the previous length of instances (or None.)

        Returns:
            tree(nested list of obj): The tree to predict unseen instances.
        """
        attribute = self.select_attribute(current_instances, available_attributes)
        available_attributes.remove(attribute)
        tree = ["Attribute", attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        all_parts_len = len(current_instances)
        partitions = self.partition_instances(current_instances, attribute)

        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            values_subtree = ["Value", att_value]
            # CASE 1
            if len(att_partition) > 0 and myutils.all_same_class(att_partition):
                values_subtree.append(["Leaf", att_partition[0][-1], len(att_partition), all_parts_len])
            # CASE 2
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                majority_class = myutils.compute_majority_vote(att_partition)
                values_subtree.append(["Leaf", majority_class, len(att_partition), all_parts_len])
            # CASE 3
            elif len(att_partition) == 0:
                majority_class = myutils.compute_majority_vote(current_instances)
                tree = ["Leaf", majority_class, all_parts_len, prev_len]
                return tree
            else: # the previous conditions were all false... recurse
                subtree = self.tdidt(att_partition, available_attributes.copy(), all_parts_len)
                values_subtree.append(subtree)
            tree.append(values_subtree)
        return tree

    def select_attribute(self, instances, attributes):
        """Uses entropy to select which attribute to split on.

        Args:
            instances(list of list of obj): The list of testing samples left.
            attributes(list of obj): The list of attributes available to split on.

        Returns:
            attribute(str): The attribute with the lowest entropy value.
        """
        e_news = []
        for att in attributes:
            e_vals = []
            for domain in self.att_domain[att]:
                e_vals.append([0, {}])
                for inst in instances:
                    if inst[self.header.index(att)] == domain:
                        e_vals[-1][0] += 1
                        try:
                            e_vals[-1][1][inst[-1]] = e_vals[-1][1][inst[-1]] + 1
                        except KeyError:
                            e_vals[-1][1][inst[-1]] = 1
            e_new = 0
            for e in e_vals:
                e_new += e[0] / len(instances) * sum(-split / e[0] * math.log(split / e[0], 2) for split in e[1].values())
            e_news.append(e_new)
        return attributes[e_news.index(min(e_news))]

    def partition_instances(self, instances, split_attribute):
        """Partitions instances based on a given attribute.

        Args:
            instances(list of list of obj): The list of testing samples to partition.
            split_attribute(str): The attribute to split on.

        Returns:
            partitions(dict): The partitioned instances as values to the split attribute
                domain as keys.
        """
        # lets use a dictionary
        partitions = {} # key (string): value (subtable)
        att_index = self.header.index(split_attribute) # e.g. 0 for level
        att_domain = self.att_domain["att" + str(att_index)] # e.g. ["Junior", "Mid", "Senior"]
        for att_value in att_domain:
            partitions[att_value] = []
        for instance in instances:
            partitions[instance[att_index]].append(instance)
        return partitions

    def tdidt_predict(self, tree, instance):
        """Traverses the tree created using TDIDT to predict the class of an instance.

        Args:
            tree(nested list of obj): The current tree to be traversed (recursive).
            instance(list of obj): The instance to be predicted.

        Returns:
            y_predicted(str): The predicted target y value.
        """
        # recursively traverse tree to make a prediction for instance
        # are we at a leaf node (base case) or attribute node?
        info_type = tree[0]
        if info_type == "Leaf":
            return tree[1]
        # we are at an attribute node
        # find attribute value match for instance
        # for loop
        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(value_list[2], instance)

    def traverse_each_path(self, tree, curr_path, all_path):
        """Traverses each path of a tree, building each path as a list.

        Args:
            tree(nested list of obj): The current tree (recursive).
            curr_path(list of obj): The current path being traversed.
            all_path(list of list of obj): All paths within the tree (result).
        """
        curr_path.append(tree[1])
        if tree[0] == "Leaf":
            all_path.append(curr_path)
            return
        else:
            for i in range(2, len(tree)):
                curr_path_copy = curr_path.copy()
                curr_path_copy.append(tree[i][1])
                self.traverse_each_path(tree[i][2], curr_path_copy, all_path)
