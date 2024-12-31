import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class InsuranceStatisticalModeling:
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y = None
        self.hgbr = None
        self.xgb = None
        self.hgbr_train_score = None
        self.hgbr_test_score = None
        self.xgb_train_score = None
        self.xgb_test_score = None

    def prepare_data(self):
        # Handle missing data
        imputer = SimpleImputer(strategy='mean')
        self.X = imputer.fit_transform(self.df[['age', 'vehicle_value'] + [col for col in self.df.columns if col.startswith('Province_') or col.startswith('PostalCode_')]])
        self.y = self.df['TotalClaims']

        # Check if the DataFrame has any valid rows
        if len(self.X) == 0 or len(self.y) == 0:
            print("The DataFrame is empty after handling missing data.")
            print("Printing the original DataFrame:")
            print(self.df)
            raise ValueError("The DataFrame is empty after handling missing data. Please check your data.")

        # Print the column names and the first few rows of the DataFrame
        print("Columns in the DataFrame:")
        print(self.df.columns)
        print("First few rows of the DataFrame:")
        print(self.df.head())

    def build_and_evaluate_models(self):
        # HistGradientBoostingRegressor
        self.hgbr = HistGradientBoostingRegressor(random_state=42)
        self.hgbr.fit(self.X, self.y)
        self.hgbr_train_score = self.hgbr.score(self.X, self.y)

        # XGBoostRegressor
        self.xgb = XGBRegressor(random_state=42)
        self.xgb.fit(self.X, self.y)
        self.xgb_train_score = self.xgb.score(self.X, self.y)

        # Train-test split and evaluate on test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        self.hgbr_test_score = self.hgbr.score(self.X_test, self.y_test)
        self.xgb_test_score = self.xgb.score(self.X_test, self.y_test)

        print(f"HistGradientBoostingRegressor Train R2 Score: {self.hgbr_train_score:.2f}")
        print(f"HistGradientBoostingRegressor Test R2 Score: {self.hgbr_test_score:.2f}")
        print(f"XGBoostRegressor Train R2 Score: {self.xgb_train_score:.2f}")
        print(f"XGBoostRegressor Test R2 Score: {self.xgb_test_score:.2f}")
