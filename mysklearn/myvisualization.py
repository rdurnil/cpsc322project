import numpy as np
import matplotlib.pyplot as plt

def scatter_chart_creation(x, y, x_label='', y_lable='', title=''):
    plt.figure()
    plt.scatter(x, y, marker="x", s=100, c="purple")
    plt.xlabel(x_label)
    plt.ylabel(y_lable)
    plt.title(title)
    plt.tight_layout() 
    plt.show()
