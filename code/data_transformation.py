import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils import object_saving


class data_process_config:

    scalear = os.path.join("arfifacts", "scaler")

# initialisation


class initialisation_config:

    def __init__(self) -> None:

        self.data_process_config = data_process_config()

    def data_transformation(self, train_data, test_data):

        try:

            # pipelines

           # numaric pipelines
            numaric_pipelines = Pipeline(

                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())

                ]
            )

            # catagorical pipelies
            catagorical_pipelines = Pipeline(

                [
                    ("ohe", OneHotEncoder()),
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler(with_mean=False))


                ]
            )

            # numaric values
            numaeics = [

                "Age"
            ]

            catagorics = [
                "Fever",
                "Cough",
                "Fatigue",
                "Difficulty Breathing",
                "Gender",
                "Blood Pressure",
                "Cholesterol Level",
            ]

            transformer = ColumnTransformer([

                ("numarics", numaric_pipelines, numaeics),

                ("catagoricals", catagorical_pipelines, catagorics)
            ])

            train_data = pd.read_csv(train_data)  # train_data
            test_data = pd.read_csv(test_data)  # test_data
            """
            splitting the data into training and testing sets for training 
            """
            # train_data_input_feature
            train_data_input_feature = train_data.iloc[:, :-1]

            # test_data_targer_feature
            train_data_targer_feature = train_data.iloc[:, -1]

            # test_data_input_feature
            test_data_input_feature = test_data.iloc[:, :-1]

            # test_data_targer_feature
            test_data_targer_feature = test_data.iloc[:, -1]

            train_arr_file = transformer.fit_transform(
                train_data_input_feature)

            """save the scaling object in the form of pickle file
            """

            object_saving(scale=transformer)

            test_arr_file = transformer.transform(test_data_input_feature)

            """ concatenate the features
            """

            """
            transforming the target variables
            """
            my_dict = {

                "Positive": 1,
                "Negative": 0

            }

            train_arr_file_ = train_data_targer_feature.map(
                my_dict)
            test_arr_file_ = test_data_targer_feature.map(
                my_dict)

            train_arr = np.c_[
                train_arr_file, np.array(train_arr_file_)]

            test_arr = np.c_[
                test_arr_file, np.array(test_arr_file_)]

            return (

                train_arr,
                test_arr,

            )

        except Exception as e:
            raise e
