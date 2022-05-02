import numpy as np
import mysklearn.myutils as myutils
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myrandomforestclassifier import MyRandomForestClassifier

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

if __name__ == "__main__":
    np.random.seed(0)
    interview_classifier = MyRandomForestClassifier(5, 3, 2)
    interview_classifier.fit(X_train, y_train)
    for tree in interview_classifier.classifier_forest:
        tree.print_decision_rules()
        print()
    print(interview_classifier.predict(["Junior", "Python", "yes", "yes"]))

