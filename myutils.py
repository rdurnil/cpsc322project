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
