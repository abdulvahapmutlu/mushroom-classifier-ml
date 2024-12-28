import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# Load data and XGBoost model
X_test, y_test = joblib.load("models/data_splits.pkl")[1:3]
xgb_model = joblib.load("models/trained_model_XGBoost.pkl")

# SHAP analysis
dtest = xgb.DMatrix(X_test)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(dtest)

# Summary plot
shap.summary_plot(shap_values, X_test)
plt.savefig("outputs/shap_summary_plots/shap_summary_plot.png")
