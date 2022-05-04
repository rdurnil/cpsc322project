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

interview_classifier = MyRandomForestClassifier(7, 3, 2)
interview_classifier.fit(X_train, y_train)

def test_myrandomforestclassifier_fit():
    # with seed 0
    best_trees = [['Attribute', 'att0', 
                    ['Value', 'Junior', 
                        ['Leaf', 'True', 3, 14]
                    ], 
                    ['Value', 'Mid', 
                        ['Leaf', 'True', 4, 14]
                    ], 
                    ['Value', 'Senior', 
                        ['Attribute', 'att2', 
                            ['Value', 'no', 
                                ['Leaf', 'False', 5, 7]
                            ], 
                            ['Value', 'yes', 
                                ['Leaf', 'True', 2, 7]
                            ]
                        ]
                    ]
                    ],
                    ['Attribute', 'att3', 
                        ['Value', 'no', 
                            ['Leaf', 'True', 10, 14]
                        ], 
                        ['Value', 'yes', 
                            ['Attribute', 'att0', 
                                ['Value', 'Junior', 
                                    ['Leaf', 'False', 1, 4]
                                ], 
                                ['Value', 'Mid', 
                                    ['Leaf', 'True', 2, 4]
                                ], 
                                ['Value', 'Senior', 
                                    ['Leaf', 'True', 1, 4]
                                ]
                            ]
                        ]
                    ],
                    ['Attribute', 'att0', 
                        ['Value', 'Junior', 
                            ['Leaf', 'True', 5, 14]
                        ], 
                        ['Value', 'Mid', 
                            ['Leaf', 'True', 2, 14]
                        ], 
                        ['Value', 'Senior', 
                            ['Attribute', 'att2', 
                                ['Value', 'no', 
                                    ['Leaf', 'False', 6, 7]
                                ], 
                                ['Value', 'yes', 
                                    ['Leaf', 'True', 1, 7]
                                ]
                            ]
                        ]
                    ]
                ]
    for tree in interview_classifier.classifier_forest:
        assert tree.tree in best_trees

def test_myrandomforestclassifier_predict():
    X_test = [["Senior", "R", "no", "no"], ["Mid", "Python", "yes", "no"], ["Junior", "R", "no", "yes"]]
    y_pred_desk_calc = ["False", "True", "True"]
    y_pred = interview_classifier.predict(X_test)
    assert y_pred == y_pred_desk_calc

    

