import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_transformation import initialisation_config
import os
from dataclasses import dataclass
from prediction import prediction


@dataclass
class data_ingestion_config:

    ROW_DATA = os.path.join("artifacts", 'Row_data.csv')

    TRAIN_DATA = os.path.join("artifacts", 'Train_data.csv')

    TEST_DATA = os.path.join("artifacts", 'Test_data.csv')


class data_ingestion_init:
    def __init__(self) -> None:

        # initialistion
        self.data_ingestion_config = data_ingestion_config()

    def data_ingestion(self):

        # data frame
        data = pd.read_csv(
            "D:\Projects\human disease prediction\data\Disease_symptom_and_patient_profile_dataset.csv")

        # row path
        os.makedirs(os.path.dirname(
            self.data_ingestion_config.ROW_DATA), exist_ok=True)

        # row csv file
        data.to_csv(data_ingestion_config.ROW_DATA, header=True, index=False)

        # splitting hte data into training and testing
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=1)

        # train path
        os.makedirs(os.path.dirname(
            self.data_ingestion_config.TRAIN_DATA), exist_ok=True)

        # train csv file
        train_data.to_csv(data_ingestion_config.TRAIN_DATA,
                          header=True, index=False)

        # test path
        os.makedirs(os.path.dirname(
            self.data_ingestion_config.TEST_DATA), exist_ok=True)

        # test csv file
        test_data.to_csv(data_ingestion_config.TEST_DATA,
                         header=True, index=False)

        return (


            self.data_ingestion_config.TRAIN_DATA,
            self.data_ingestion_config.TEST_DATA

        )


if __name__ == "__main__":
    obj = data_ingestion_init()
    print("data ingestion initialization")

    train_data, test_data = obj.data_ingestion()
    print("getting the training data and test data")

    initialisation_config_ = initialisation_config()

    train_arr, test_arr = initialisation_config_.data_transformation(
        train_data=train_data, test_data=test_data)
    print("transformed arrys are getting the training data and test data")

    # prediction object
    pred_obj = prediction()

    print("suitable algorithm are getting in the form of pickle")
    pred_obj.predict(train_data=train_arr, test_data=test_arr)


print("successful")
