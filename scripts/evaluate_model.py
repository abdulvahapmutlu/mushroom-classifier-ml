import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and models
X_test, y_test = joblib.load("models/data_splits.pkl")[1:3]
model_files = ["models/trained_model_RandomForest.pkl", "models/trained_model_GradientBoosting.pkl",
               "models/trained_model_XGBoost.pkl", "models/trained_model_ExtraTrees.pkl", "models/trained_model_SVC.pkl"]

# Evaluate models
for model_file in model_files:
    model_name = model_file.split("_")[-1].replace(".pkl", "")
    model = joblib.load(model_file)
    print(f"Evaluating {model_name}...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
