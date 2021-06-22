import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

class Trainer():

    def load_data(self):
        """
        load the data and return X and y
        """

        # read data
        data = pd.read_csv("../data/movie_popularity.csv")

        # clean data
        data = data.drop_duplicates()
        data = data.drop("revenue", axis=1)
        data.dropna(inplace=True)

        # extract target
        y = data.popularity
        X = data.drop("popularity", axis=1)

        return X, y
    


    def create_pipeline(self):
        
        numerical_features  = ['duration_min','budget','number_of_awards_won','number_of_nominations','has_collection','number_of_top_productions']
        categorical_features  = ["original_language", "status", "available_in_english"]

                               
        preproc = ColumnTransformer([
        ("numerical_scaler", RobustScaler(), numerical_features),
        ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_features)
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
        
        print(full.score(X,y))
        
        # save pipeline
        joblib.dump(full, "../model.joblib")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()