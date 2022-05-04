from mysklearn.mypytable import MyPyTable
from mysklearn import myutils
from mysklearn import myevaluation
from mysklearn.myrandomforestclassifier import MyRandomForestClassifier
from mysklearn.myclassifiers import MyDecisionTreeClassifier

def interview():
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
            ["Senior", "Java", "no", "no", "False"],
            ["Senior", "Java", "no", "yes", "False"],
            ["Mid", "Python", "no", "no", "True"],
            ["Junior", "Python", "no", "no", "True"],
            ["Junior", "R", "yes", "no", "True"],
            ["Junior", "R", "yes", "yes", "False"],
            ["Mid", "R", "yes", "yes", "True"],
            ["Senior", "Python", "no", "no", "False"],
            ["Senior", "R", "yes", "no", "True"],
            ["Junior", "Python", "yes", "no", "True"],
            ["Senior", "Python", "yes", "yes", "True"],
            ["Mid", "Python", "no", "yes", "True"],
            ["Mid", "Java", "yes", "no", "True"],
            ["Junior", "Python", "no", "yes", "False"]
        ]
    interview_test_x = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]

    interview_data = MyPyTable(interview_header, interview_table)
    interview_att = interview_data.get_multiple_columns(["level", "lang", "tweets", "phd"], "")
    interview_class = interview_data.get_column("interviewed_well")
    forest = MyRandomForestClassifier(n_val=5, m_val=2, f_val=2)
    forest.fit(interview_att.data, interview_class)
    for _, tree in enumerate(forest.classifier_forest):
        print(tree.tree)
    print(forest.predict(interview_test_x))


def big_test():

    discretized_data = MyPyTable()
    discretized_data.load_from_file("meteorological-dataset.csv")
    discretized_names = ["Month","Temperature","Relative Humidity","Solar Radiation","Precipitaion","Windspeed","Wind Direction","Shelter Temperature"]
    discretized_att_data = discretized_data.get_multiple_columns(discretized_names, "", False)
    discretized_class_data = discretized_data.get_column("Ozone Level")

    train_folds, test_folds = myevaluation.stratified_kfold_cross_validation(discretized_att_data.data, discretized_class_data, n_splits=10, random_state=None, shuffle=True)

    discretized_att_train_data = myutils.create_empty_list(len(train_folds))
    discretized_att_test_data = myutils.create_empty_list(len(train_folds))
    discretized_class_train_data = myutils.create_empty_list(len(train_folds))
    discretized_class_test_data = myutils.create_empty_list(len(train_folds))

    for i,_ in enumerate(train_folds):
        discretized_att_train_data[i] = myutils.indexes_to_values(train_folds[i], discretized_att_data.data)
        discretized_att_test_data[i] = myutils.indexes_to_values(test_folds[i], discretized_att_data.data)
        discretized_class_train_data[i] = myutils.indexes_to_values(train_folds[i], discretized_class_data)
        discretized_class_test_data[i] = myutils.indexes_to_values(test_folds[i], discretized_class_data)

    discretized_class_labels = myutils.find_unique_items(discretized_class_data)

    rf_y_pred, rf_y_true = myutils.perform_classification_on_folds(MyRandomForestClassifier(), discretized_att_train_data, discretized_class_train_data, discretized_att_test_data, discretized_class_test_data)
    print("Random Forest Classifier Analysis: ")
    myutils.perform_analysis_on_classification(rf_y_true, rf_y_pred, discretized_class_labels, ["Ozone Group"])

if __name__ == "__main__":
    #interview()
    big_test()