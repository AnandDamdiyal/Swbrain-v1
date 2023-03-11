import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class BrainDataset:
    def __init__(self, data_file):
        self.data_file = data_file
    
    def load_data(self):
        # Load the data from the CSV file
        df = pd.read_csv(self.data_file)
        
        # Split the data into input features (X) and labels (y)
        X = df.drop(columns=['label']).to_numpy()
        y = df['label'].to_numpy()
        
        # Standardize the input features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
