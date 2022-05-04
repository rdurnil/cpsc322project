
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
    if float(temp) <= -5:
        assigned_class = "< -5"
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
    if float(humidity) <= 20:
        assigned_class = "< 20"
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
    if float(value) <= 2:
        assigned_class = "< 2"
    elif float(value) <= 4:
        assigned_class = "2 - 4"
    elif float(value) <= 6:
        assigned_class = "4 - 6"
    elif float(value) <= 8:
        assigned_class = "6 - 8"
    else:
        assigned_class = "> 8"
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
    else:
        assigned_class = "> 4"
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
    return assigned_class

def wind_speed(value):
    if float(value) <= 1:
        assigned_class = "< 1"
    elif float(value) <= 2:
        assigned_class = "1 - 2"
    elif float(value) <= 3:
        assigned_class = "2 - 3"
    else:
        assigned_class = "> 3"
    return assigned_class

def shelter_temp(value):
    if float(value) <= 20:
        assigned_class = "< 20"
    elif float(value) <= 22:
        assigned_class = "20 - 22"
    elif float(value) <= 24:
        assigned_class = "22 - 24"
    elif float(value) <= 26:
        assigned_class = "24 - 26"
    elif float(value) <= 28:
        assigned_class = "26 - 28"
    else:
        assigned_class = "> 28"
    return assigned_class

def date_time(value):
    month = str(value)[0:2]
    if month == "01":
        assigned_class = "January"
    elif month == "02":
        assigned_class = "February"
    elif month == "03":
        assigned_class = "March"
    elif month == "04":
        assigned_class = "April"
    elif month == "05":
        assigned_class = "May"
    elif month == "06":
        assigned_class = "June"
    elif month == "07":
        assigned_class = "July"
    elif month == "08":
        assigned_class = "August"
    elif month == "09":
        assigned_class = "September"
    elif month == "10":
        assigned_class = "October"
    elif month == "11":
        assigned_class = "November"
    elif month == "12":
        assigned_class = "December"
    return assigned_class
