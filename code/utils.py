import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

"""
save the scale feature as pkl file"""


def object_saving(scale):
    try:
        path_file = os.path.join("artifacts", "scaling.pkl")

        # make a directory for the scaling objects
        os.makedirs(os.path.dirname(path_file), exist_ok=True)

        with open(path_file, "wb") as f:

            pickle.dump(scale, f)

    except Exception as e:
        raise e


def predictions(path_file):
    pass

    """
        apply the algorithm and gridsearch to the model and return the result as a dictionary"""


def processing(train_df, train_df_target, test_df, test_df_target, algorithms_, parameters):

    try:

        my_dict = {}

        for i in range(len(list(algorithms_))):

            algorithms = list(algorithms_.values())[i]

            params = parameters[list(algorithms_.keys())[i]]

            gs = GridSearchCV(algorithms, params, cv=3)

            gs.fit(train_df, train_df_target)

            algorithms.set_params(**gs.best_params_)

            algorithms.fit(train_df, train_df_target)

            y_train_pred = algorithms.predict(train_df)

            y_test_pred = algorithms.predict(test_df)

            train_model_score = r2_score(train_df_target, y_train_pred)

            test_model_score = r2_score(test_df_target, y_test_pred)

            my_dict[list(algorithms_.keys())[i]] = test_model_score

        final_algorithm = max(list(my_dict))
        return final_algorithm

    except Exception as e:

        raise e


"""
saving the algorithm to a file"""


def algorithm_saving(algo, path):
    try:
        # make a directory for the scaling objects
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:

            pickle.dump(algo, f)

    except Exception as e:
        raise e
