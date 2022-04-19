from mypytable import MyPyTable
import matplotlib.pyplot as plt
import myevaluation

def ozone_assigning(ozone_value):
    """Classifies a mpg list into the levels
        Args:
            mpg_list (list): the list of mpg data
        Returns:
            A list of the classified values
    """
    if float(ozone_value) >= 36.0:
        assigned_class = "High"
    elif float(ozone_value) >= 16.0:
        assigned_class = "Normal"
    else:
        assigned_class = "Low"
    return assigned_class

def bar_chart(x, y, title="", xlabel="", ylabel="", rotation=0):
    """Creates a bar chart.
    
    Args:
        x (list): list of values to be plotted on the x-axis.
        y (list): parallel to x, the height of each corresponding bar.
        title (string): title for the graph. (optional)
        xlabel (string): label for the x-axis. (optional)
        ylabel (string): label for the y-axis. (optional)
        rotation (int): rotation for the xlabels. (optional)
    """
    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, rotation=rotation)
    plt.grid()
    plt.show()

def find_unique_items(item_list):
    """Finds the unique items in a list
        Args:
            item_list (list of objs): list of objects to search
        Returns:
            a sorted list of the unique items
    """
    unique_list = []
    for _, row in enumerate(item_list):
        if row not in unique_list:
            unique_list.append(row)
    unique_list.sort()
    return unique_list

def count_unique_items(data_set, unique_items):
    """Counts the number of unique items in a 1D list
        Args:
            data_set (list): the list being searched
            unique_items (list): the items being searched for
        Returns:
            A list of the number of each unique item
    """
    count_list = []
    for i, _ in enumerate(unique_items):
        num_this_item = 0
        for j,_ in enumerate(data_set):
            if unique_items[i] == data_set[j]:
                num_this_item += 1
        count_list.append(num_this_item)
    return count_list

def perform_classification_on_folds(classifier, x_train, y_train, x_test, y_test):
    """Classifies the data using the given datasets
        Args:
            classifier (obj): the classifier to predict with
            x_train (list of obj): the training attributes
            y_train (list of obj): the training classes
            x_test (list of obj): the testing attributes
            y_test (list of obj): the testing classes
        Returns:
            y_pred (list of obj): the classes predicted by the classifier
            y_true (list of obj): the expected classes
    """
    y_pred = []
    y_true = []
    for i, _ in enumerate(x_train):
        classifier.fit(x_train[i], y_train[i])
        temp_y_pred = classifier.predict(x_test[i])
        y_pred.extend(temp_y_pred)
        y_true.extend(y_test[i])
    return y_pred, y_true

def perform_analysis_on_classification(y_true, y_pred, unique_labels, matrix_title, pos_label=None):
    # matrix_title = ["Team","H", "A", "Total", "Recongnition (%)"]
    # This is an example of what the title needs to look like for the confusion matrix
    # the H and A are the unique values in the lables. Normally I just hard-coded them 
    # but we probably need a code solution 
    # This would probably work:
    #   matrix_title = ["Ozone Classification"]
    #   matirx_title.extend(unique_labels)
    #   matrix_title.extend(["Total", "Recongnition (%)"]) 
    # This will probabily go prior to passing in the values 
    """Calculates the values neccesary to determine if the classifier is working correctly
        Args:
            y_true (list of obj): the expected values for the class
            y_pred (list of obj): the values predicted by the classifier
            unique_labels (list of obj): the possible results for the y values
            matrix_title (list of obj): the title that goes at the top of the tabulated table
            pos_label (obj): the object from the unique labels that is the positive classification
                if none, it will be the first item in the unique_labels
    """
    print("Accuracy: ", myevaluation.accuracy_score(y_true, y_pred, normalize=True))
    print("Error Rate: ", myevaluation.error_rate(y_true, y_pred, normalize=True))
    print("Precision: ", myevaluation.binary_precision_score(y_true, y_pred, unique_labels, pos_label))
    print("Recall: ", myevaluation.binary_recall_score(y_true, y_pred, unique_labels, pos_label))
    print("F1 measure: ", myevaluation.binary_f1_score(y_true, y_pred, unique_labels, pos_label))
    print("Confusion Matrix")
    print("-------------------------------")
    print(myevaluation.create_matrix_table(y_true, y_pred, unique_labels, matrix_title))
