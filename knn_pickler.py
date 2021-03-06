import pickle 
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyKNeighborsClassifier

ozone_table = MyPyTable()
ozone_table.load_from_file("meteorological-hourly-clean.csv")
y_data = ozone_table.get_column("OZONE_LEVEL")
X_data_table = ozone_table.get_multiple_columns(["TEMPERATURE","RELATIVE_HUMIDITY","SOLAR_RADIATION",\
    "PRECIPITATION","WINDSPEED","WIND_DIRECTION","SHELTER_TEMPERATURE"], "")

knn_classifier = MyKNeighborsClassifier()
knn_classifier.fit(X_data_table.data, y_data)

packaged_obj = knn_classifier
outfile = open("trained_knn.p", "wb") # binary
pickle.dump(packaged_obj, outfile)
outfile.close()