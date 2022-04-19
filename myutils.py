from mypytable import MyPyTable

def ozone_assigning(ozone_value):
    """Classifies a mpg list into the levels
        Args:
            mpg_list (list): the list of mpg data
        Returns:
            A list of the classified values
    """
    if float(ozone_value) >= 45.0:
        assigned_number = 10
    elif float(ozone_value) >= 37.0:
        assigned_number = 9
    elif float(ozone_value) >= 31.0:
        assigned_number = 8
    elif float(ozone_value) >= 27.0:
        assigned_number = 7
    elif float(ozone_value) >= 24.0:
        assigned_number = 6
    elif float(ozone_value) >= 20.0:
        assigned_number = 5
    elif float(ozone_value) >= 17.0:
        assigned_number = 4
    elif float(ozone_value) >= 15.0:
        assigned_number = 3
    elif float(ozone_value) > 13.0:
        assigned_number = 2
    else:
        assigned_number = 1

    return assigned_number