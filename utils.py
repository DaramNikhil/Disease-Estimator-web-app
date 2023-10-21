import pandas as pd
import pickle
import os
from sklearn.metrics import r2_score


def object_saving(scale):
    try:
        path_file = os.path.join("artifacts", "scaling.pkl")

        # make a directory for the scaling objects
        os.makedirs(os.path.dirname(path_file), exist_ok=True)

        with open(path_file, "wb") as f:

            pickle.dump(scale, f)

    except Exception as e:
        raise e
