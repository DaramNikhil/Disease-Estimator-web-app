import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils import processing
import os
from utils import predictions
from utils import algorithm_saving
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


class prediction_config:

    """
    file path of saving object file"""

    pred_file_path = os.path.join("artifacts", "algorithm.pkl")


class prediction:

    def __init__(self):
        self.config = prediction_config()

    """
    creating the prediction configuration for a given classifier class
    """

    def predict(self, train_data, test_data):

        try:

            """
            creating a dictionary of the algorithms
            """
            algorithms_ = {'RandomForestClassifier': RandomForestClassifier(),
                           "SVC": SVC(),
                           "GaussianNB": GaussianNB(),
                           "KNN": KNeighborsClassifier(n_neighbors=5),
                           "DecisionTreeClassifier": DecisionTreeClassifier()
                           }

            """
            split the trainig and test data"""

            train_df = train_data[:, :-1]
            train_df_target = train_data[:, -1]
            test_df = test_data[:, :-1]
            test_df_target = test_data[:, -1]

            os.makedirs(os.path.dirname(
                self.config.pred_file_path), exist_ok=True)

            """
            hyper parameter tuning parameters"""
            parameters = {
                "RandomForestClassifier": {"max_depth": [2, 3, 5, 10, 20],
                                           "criterion": ["gini", "entropy"],
                                           "max_features": ["sqrt", "log2"],
                                           "min_samples_split": [2, 5, 10],
                                           "min_samples_leaf": [1, 2, 4]
                                           },

                "SVC": {"C": [0.1, 1, 10, 100, 1000],
                        "kernel": ["rbf", "linear"],
                        "gamma": [1, 0.1, 0.01, 0.001, 0.0001]
                        },

                "GaussianNB": {
                },


                "DecisionTreeClassifier": {"min_samples_split": [2, 5, 10],
                                           "min_samples_leaf": [1, 2, 4]
                                           },

                "KNN": {

                }
            }

            """
            apply the algorithm and gridsearch to the model and return the result as a dictionary"""

            best_algorithm = processing(

                train_df=train_df,
                train_df_target=train_df_target,
                test_df=test_df,
                test_df_target=test_df_target,
                algorithms_=algorithms_,
                parameters=parameters,


            )

            main_algorithm = algorithms_[str(best_algorithm)]

            main_algorithm.fit(train_df, train_df_target)

            algorithm_saving(algo=main_algorithm,
                             path=self.config.pred_file_path)

            prediction = main_algorithm.predict(test_df)

            model_scores = r2_score(test_df_target, prediction)

            return self.config.pred_file_path

        except Exception as e:

            raise e
