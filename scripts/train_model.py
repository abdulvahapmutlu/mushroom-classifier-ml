import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn.svm import SVC

# Load data
X_train, X_test, y_train, y_test = joblib.load("models/data_splits.pkl")

# Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42),
    "SVC": SVC(C=1, kernel="rbf", probability=True, random_state=42)
}

# Train models and save
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/trained_model_{model_name}.pkl")
    print(f"{model_name} saved.")
