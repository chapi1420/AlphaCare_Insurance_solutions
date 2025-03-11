import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

class FeatureEngineering:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)

    def clean_data(self):
        # Replace empty strings and 'Unknown' with NaN
        self.data.replace(['', 'Unknown'], np.nan, inplace=True)
        
        # Fill numeric NaNs with median and categorical NaNs with mode
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col] = self.data[col].fillna(self.data[col].median())
        
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

    def process_dates(self):
        if 'VehicleIntroDate' in self.data.columns:
            self.data['VehicleIntroDate'] = pd.to_datetime(self.data['VehicleIntroDate'], format='%Y-%m-%d', errors='coerce')
            self.data['VehicleIntroYear'] = self.data['VehicleIntroDate'].dt.year
            self.data['VehicleAge'] = 2025 - self.data['VehicleIntroYear']

    def engineer_features(self):
        if 'CapitalOutstanding' in self.data.columns and 'SumInsured' in self.data.columns:
            self.data['CapitalOutstanding'] = pd.to_numeric(self.data['CapitalOutstanding'], errors='coerce')
            self.data['SumInsured'] = pd.to_numeric(self.data['SumInsured'], errors='coerce')
            self.data['OutstandingRatio'] = self.data['CapitalOutstanding'] / (self.data['SumInsured'] + 1)

    def simplify_categories(self, threshold=0.01):
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            freq = self.data[col].value_counts(normalize=True)
            rare_labels = freq[freq < threshold].index
            self.data[col] = self.data[col].apply(lambda x: 'Other' if x in rare_labels else x)

    def encode_categorical(self):
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].nunique() < 10:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(self.data[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
                self.data = pd.concat([self.data, encoded_df], axis=1)
                self.data.drop(col, axis=1, inplace=True)
            else:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])

    def scale_numerical(self):
        scaler = StandardScaler()
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

    def balance_data(self, target_column):
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        self.data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_column)], axis=1)

    def drop_irrelevant_columns(self, columns_to_drop):
        self.data.drop(columns=columns_to_drop, axis=1, inplace=True)

    def process(self, target_column):
        self.clean_data()
        self.process_dates()
        self.engineer_features()
        self.simplify_categories()
        self.encode_categorical()
        self.scale_numerical()
        self.balance_data(target_column)
        self.drop_irrelevant_columns(['VehicleIntroDate'])
        return self.data

if __name__ == "__main__":
    processor = FeatureEngineering('/home/nahomnadew/Desktop/10x/AlphhaCare_Insurance-v2/AlphaCare_Insurance_solutions/Data/MachineLearningRating_v3.1.csv')
    processed_data = processor.process(target_column='TargetColumn')
    processed_data.to_csv('processed_data.csv', index=False)
    print("Data processing complete.")