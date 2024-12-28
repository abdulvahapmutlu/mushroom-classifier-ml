# MushroomClassifier: Classifying Mushrooms with Advanced Machine Learning Techniques

## Overview

**MushroomClassifier** is an end-to-end machine learning pipeline for identifying whether a mushroom is **edible** or **poisonous** based on its physical characteristics. Using a diverse dataset and state-of-the-art machine learning techniques, this project demonstrates best practices in preprocessing, training, evaluation, and model explainability.

The goal is to provide an accurate, interpretable, and reproducible classification system to identify mushrooms' edibility based on their observable attributes.

---

## Features

- **Data Preprocessing:** Handles missing values, encodes categorical features, and scales numerical variables.
- **Multi-Model Comparison:** Includes advanced classification algorithms:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Extra Trees
  - Support Vector Classifier (SVC)
- **Performance Metrics:** Evaluates models using:
  - Precision, recall, and F1-score
  - Confusion matrices
  - ROC-AUC scores
  - Cross-validation metrics
- **Explainability with SHAP:** Provides insights into how each feature impacts the predictions, making the results interpretable and actionable.
- **Reproducibility:** A structured workflow with modular components ensures ease of use and reproducibility.

---

## Installation

### Clone the Repository

```
git clone https://github.com/abdulvahapmutlu/mushroom-classifier-ml.git
cd MushroomClassifier
```

### Install Dependencies

Create a virtual environment (optional but recommended):

```
python -m venv env
source env/bin/activate  # For Linux/Mac
env\Scripts\activate     # For Windows
```

Install required Python packages:

```
pip install -r requirements.txt
```

---

## Usage Guide

### 1. **Preprocessing**
   - Prepare the dataset for training by executing the `scripts/preprocessing.py` script:
     ```
     python scripts/preprocessing.py
     ```

### 2. **Model Training**
   - Train and compare machine learning models with the `scripts/train_model.py` script:
     ```
     python scripts/train_model.py
     ```

### 3. **Evaluation**
   - Evaluate model performance on the mushroom dataset using the evaluation script:
     ```
     python scripts/evaluate_model.py
     ```

### 4. **SHAP Analysis**
   - Understand the model's decisions by running the SHAP analysis  `scripts/shap_analysis.py` script:
     ```
     python scripts/shap_analysis.py
     ```

---

## Outputs

The following outputs are generated and stored in the `outputs/` directory:

1. **Confusion Matrices:**
   - Located in `outputs/`.
   - Visual comparisons of true and predicted classifications.

2. **Learning Curves:**
   - Stored in `outputs/`.
   - Plots of training and validation loss over iterations.

3. **SHAP Summary Plots:**
   - Found in `outputs/`.
   - Visualizations highlighting the most influential features in the mushroom classification task.

---

## Dependencies

The project requires the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `shap`
- `matplotlib`
- `seaborn`

Refer to the `requirements.txt` file for a complete list.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For any questions or feedback, feel free to contact:

- **Abdulvahap Mutlu**
- Email: [abdulvahapmutlu1@gmail.com](mailto:abdulvahapmutlu1@gmail.com)
