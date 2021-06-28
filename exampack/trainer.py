
import pandas as pd

import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class Trainer():

    def load_data(self):
        """
        load the data and return X and y
        """

        # read data
        data_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "movie_popularity.csv")
        data = pd.read_csv(data_path)

        # clean data
        data = data.drop_duplicates()
        data = data.drop("revenue", axis=1)
        data.dropna(inplace=True)

        # extract target
        y = data.popularity
        X = data.drop("popularity", axis=1)

        return X, y

    def create_pipeline(self):
        """
        the pipeline expects to be trained with a DataFrame containing
        the following data types in that order
        ```
        original_title              string
        title                       string
        release_date                string
        duration_min                float
        description                 string
        budget                      float
        original_language           string
        status                      string
        number_of_awards_won        int
        number_of_nominations       int
        has_collection              int
        all_genres                  string
        top_countries               string
        number_of_top_productions   float
        available_in_english        bool
        ```
        """

        numerical_features = [
            "duration_min",
            "budget",
            "number_of_awards_won",
            "number_of_nominations",
            "has_collection",
            "number_of_top_productions"]

        categorical_features = [
            "original_language",
            "status",
            "available_in_english"]

        preproc = ColumnTransformer([
            ("numerical_scaler", RobustScaler(), numerical_features),
            ("one_hot_encoding", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_features)
        ])

        return preproc

    def train(self):
        """
        load the data and train a pipelined model
        the pipelined model is saved to model.joblib
        """

        # load data
        X, y = self.load_data()

        # create pipeline
        pipeline = self.create_pipeline()
        model = Lasso()
        full = make_pipeline(pipeline, model)

        # fit pipeline
        full.fit(X, y)

        print(full.score(X, y))

        # save pipeline
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "model.joblib")
        joblib.dump(full, model_path)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
