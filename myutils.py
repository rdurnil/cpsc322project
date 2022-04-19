from mypytable import MyPyTable
import matplotlib.pyplot as plt

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
