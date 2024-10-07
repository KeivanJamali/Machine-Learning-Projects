import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class FlowDataLoader:
    def __init__(self, base_path, city, random, missing_rate):
        """
        Initialize the data loader with paths to data.
        Arguments:
        base_path : str : The base path to the dataset folder.
        city : str : The city name (e.g., 'Anaheim', 'Sioux Falls', 'Chicago').
        missing_rate : int : The missing data percentage (e.g., 10, 20, etc.).
        """
        self.base_path = base_path
        self.city = city
        self.missing_rate = missing_rate
        self.random = random
        self.load_data()

    def load_matrix(self, file_name):
        """Helper function to load a matrix from CSV."""
        file_path = f"D:/All Python/Machine-Learning-Projects/Project 2 - Transportation/Main Project/{self.base_path}/{self.city}/Scaled/random_{self.random}/{self.missing_rate}/{file_name}"
        return pd.read_csv(file_path, index_col=0)

    def load_data(self):
        """Load and preprocess the data."""
        # Load all the necessary matrices
        self.train_od_matrix = self.load_matrix(f'at_miss0.{self.missing_rate}_train_od_matrix.csv')
        self.val_od_matrix = self.load_matrix(f'at_miss0.{self.missing_rate}_val_od_matrix.csv')
        self.test_od_matrix = self.load_matrix(f'at_miss0.{self.missing_rate}_test_od_matrix.csv')
        self.real_od_matrix = self.load_matrix('real_od_matrix.csv')
        self.attraction = self.load_matrix('attraction.csv')
        self.production = self.load_matrix('production.csv')
        self.travel_time = self.load_matrix('travel_time_matrix.csv')

    def prepare_features(self, od_matrix):
        """Prepare the feature matrix by combining production, attraction, and travel time."""
        num_zones = od_matrix.shape[0]
        features = []

        # Create a feature matrix for each OD pair
        for i in range(num_zones):
            for j in range(num_zones):
                production_i = self.production.iloc[i, 0]
                attraction_j = self.attraction.iloc[j, 0]
                travel_time_ij = self.travel_time.iloc[i, j]
                    
                # Combine features into a list [production_i, attraction_j, travel_time_ij]
                features.append([production_i, attraction_j, travel_time_ij])
        
        return np.array(features)

    def get_data(self):
        """Return the training, validation, and test sets for model training."""
        # Prepare features for train, val, and test datasets
        x_train = self.prepare_features(self.train_od_matrix)
        x_val = self.prepare_features(self.val_od_matrix)
        x_test = self.prepare_features(self.test_od_matrix)
        
        # The target is the OD flow from the real OD matrix
        y_train = self.train_od_matrix.values.flatten()
        y_val = self.val_od_matrix.values.flatten()
        y_test = self.test_od_matrix.values.flatten()
        
        # Filter out 'No_connection' (self-loops or missing data points)
        non_connection_train = (y_train != 'No_connection') & (y_train != 'False')
        non_connection_val = (y_val != 'No_connection') & (y_val != 'False')
        non_connection_test = (y_test != 'No_connection') & (y_test != 'False')

        # Convert to numeric and filter out 'No_connection' and 'False'
        y_train = y_train[non_connection_train].astype(float)
        x_train = x_train[non_connection_train]

        y_val = y_val[non_connection_val].astype(float)
        x_val = x_val[non_connection_val]

        y_test = y_test[non_connection_test].astype(float)
        x_test = x_test[non_connection_test]

        return x_train, y_train, x_val, y_val, x_test, y_test
    


# Random Forest Model Class
class RandomForestFlowPredictor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def train(self, x_train, y_train):
        # Training the Random Forest model
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        # Making predictions on test data
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        # Evaluate the model using standard metrics
        predictions = self.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return rmse, mae, r2

