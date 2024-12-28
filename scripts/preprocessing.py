import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Preprocess the data
def preprocess_data(data):
    # Separate features and target
    X = data.drop(columns=['target'])
    y = data['target']

    # Define column types
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Create preprocessing pipeline
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, X, y

# Split data
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Main script
if __name__ == "__main__":
    dataset_path = "path/to/your/dataset.csv"  # Update with actual dataset path
    data = load_data(dataset_path)
    preprocessor, X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save preprocessor and split data
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    joblib.dump((X_train, X_test, y_train, y_test), "models/data_splits.pkl")

    print("Preprocessing completed and files saved.")
