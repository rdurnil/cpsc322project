import numpy as np
import mysklearn.myutils as myutils

header = ["level", "lang", "tweets", "phd"]
attribute_domains = {"level": ["Senior", "Mid", "Junior"], 
        "lang": ["R", "Python", "Java"],
        "tweets": ["yes", "no"], 
        "phd": ["yes", "no"]}
X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

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
        entropy_calc = calculate_entropy(instances, header.index(att))
        entropy_options.append(entropy_calc)
    att_index = myutils.find_smallest_value(entropy_options)
    return attributes[att_index]

def calculate_entropy(self, instances, att_col, curr_y_train):
    """Calculates the entropy of a single column
        Args:
            instances (list of list): the rows of instances
            att_col (int): the column to calculate the entropy of
        Returns:
            The weighted sum of the column
    """
    unique_class_vals = myutils.find_unique_items(curr_y_train)
    num_class_options = len(unique_class_vals)
    entropy_total = 0
    for _, value in enumerate(attribute_domains[att_col]):
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
                entropy_calc += -proportion * np.log2(proportion)
        entropy_total += (sum(total_class_options) / len(self.y_train)) * entropy_calc
    return entropy_total

def compute_random_subset(values):
    values_copy = values[:]
    subset = set()
    if len(values_copy) < 2:
        return values_copy
    else:
        while(len(subset) < 2):
            subset.add((values_copy[np.random.randint(0, len(values_copy))]))
        return list(subset)

if __name__ == "__main__":
    np.random.seed(0)
    # M = 5, N = 3, F = 2
    print(compute_random_subset(header))
