
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

def temp_disc(temp):
    if float(temp) <= -10:
        assigned_class = "< -10"
    elif float(temp) <= -5:
        assigned_class = "-10 - -5"
    elif float(temp) <= 0:
        assigned_class = "-5 - 0"
    elif float(temp) <= 5:
        assigned_class = "0 - 5"
    elif float(temp) <= 10:
        assigned_class = "5 - 10"
    elif float(temp) <= 15:
        assigned_class = "10 - 15"
    elif float(temp) <= 20:
        assigned_class = "15 - 20"
    elif float(temp) <= 25:
        assigned_class = "20 - 25"
    elif float(temp) <= 30:
        assigned_class = "25 - 30"
    else:
        assigned_class = "> 30"
    return assigned_class

def humidity_disc(humidity):
    if float(humidity) <= 10:
        assigned_class = "< 10"
    elif float(humidity) <= 20:
        assigned_class = "10 - 20"
    elif float(humidity) <= 30:
        assigned_class = "20 - 30"
    elif float(humidity) <= 40:
        assigned_class = "30 - 40"
    elif float(humidity) <= 50:
        assigned_class = "40 - 50"
    elif float(humidity) <= 60:
        assigned_class = "50 - 60"
    elif float(humidity) <= 70:
        assigned_class = "60 - 70"
    elif float(humidity) <= 80:
        assigned_class = "70 - 80"
    elif float(humidity) <= 90:
        assigned_class = "80 - 90"
    else:
        assigned_class = "90 - 100"
    return assigned_class

def radiation(value):
    if float(value) <= 200:
        assigned_class = "< 200"
    elif float(value) <= 400:
        assigned_class = "200 - 400"
    elif float(value) <= 600:
        assigned_class = "400 - 600"
    elif float(value) <= 800:
        assigned_class = "600 - 800"
    elif float(value) <= 1000:
        assigned_class = "800 - 1000"
    else:
        assigned_class = "> 1000"
    return assigned_class

def precipitation(value):
    if float(value) <= 1:
        assigned_class = "< 1"
    elif float(value) <= 2:
        assigned_class = "1 - 2"
    elif float(value) <= 3:
        assigned_class = "2 - 3"
    elif float(value) <= 4:
        assigned_class = "3 - 4"
    elif float(value) <= 5:
        assigned_class = "4 - 5"
    elif float(value) <= 6:
        assigned_class = "5 - 6"
    elif float(value) <= 7:
        assigned_class = "6 - 7"
    elif float(value) <= 8:
        assigned_class = "7 - 8"
    elif float(value) <= 9:
        assigned_class = "8 - 9"
    else:
        assigned_class = "> 9"
    return assigned_class

def wind_speed(value):
    if float(value) <= 1:
        assigned_class = "< 1"
    elif float(value) <= 2:
        assigned_class = "1 - 2"
    elif float(value) <= 3:
        assigned_class = "2 - 3"
    elif float(value) <= 4:
        assigned_class = "3 - 4"
    elif float(value) <= 5:
        assigned_class = "4 - 5"
    else:
        assigned_class = "> 5"
    return assigned_class

def wind_dir(value):
    if float(value) <= 90:
        assigned_class = "0 - 90"
    elif float(value) <= 180:
        assigned_class = "90 - 180"
    elif float(value) <= 270:
        assigned_class = "180 - 270"
    else:
        assigned_class = "> 270"

def wind_speed(value):
    if float(value) <= 1:
        assigned_class = "< 1"
    elif float(value) <= 2:
        assigned_class = "1 - 2"
