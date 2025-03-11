import pandas as pd
class DataCleaner:
    """
    A class for cleaning and preprocessing the insurance data.
    """
    
    def __init__(self, df):
        self.df = df
        
    def handle_missing_values(self):
        """
        Handle missing values in the dataset.
        """
        # Check for missing values
        print(f"Missing values before cleaning:\n{self.df.isnull().sum()}")
        
        # Fill missing values with appropriate methods
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.df[column] = self.df[column].fillna('Unknown')
            elif self.df[column].dtype == 'float64' or self.df[column].dtype == 'int64':
                self.df[column] = self.df[column].fillna(self.df[column].median())
        
        print(f"Missing values after cleaning:\n{self.df.isnull().sum()}")
        
    def detect_and_handle_outliers(self):
        """
        Detect and handle outliers in the dataset.
        """
        # Detect outliers using the IQR method
        for column in self.df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            print(f"Outliers detected in column {column}: {len(outliers)}")
            
            # Handle outliers by capping them
            self.df.loc[self.df[column] < lower_bound, column] = lower_bound
            self.df.loc[self.df[column] > upper_bound, column] = upper_bound
            return self.df
        
    def preprocess_data(self):
        """
        Preprocess the data by handling missing values and outliers.
        """
        self.handle_missing_values()
        self.detect_and_handle_outliers()
        
        return self.df
data = pd.read_csv('C:\\Users\\nadew\\10x\\week3\\ACIS\\data\\cleaned_data\\cleaned_data_v4.csv')
data = DataCleaner(data)
data.handle_missing_values()
cleaned = data.detect_and_handle_outliers()
cleaned.to_csv('C:\\Users\\nadew\\10x\\week3\\ACIS\\data\\cleaned_data\\cleaned_data_v5.csv', index=False)
# The Janitor class is responsible for cleaning and preprocessing the insurance data. It includes methods for handling missing values, detecting and handling outliers, and preprocessing the data. The class takes a DataFrame as input and performs the cleaning operations on it. The handle_missing_values method fills missing values in the dataset with appropriate methods, while the detect_and_handle_outliers method detects and handles outliers using the IQR method. The preprocess_data method combines the two cleaning steps to preprocess the data. The cleaned data is then saved to a new CSV file for further analysis.    ``