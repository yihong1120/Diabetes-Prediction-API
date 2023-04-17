"""
Trains an XGBoost classifier on the Pima Indians Diabetes dataset and saves the best model to a file.

The dataset contains information about female patients at least 21 years old of Pima Indian heritage.
The task is to predict whether a patient has diabetes based on various diagnostic measurements.

Hyperparameters for the XGBoost classifier are selected using grid search with cross-validation.

The best model is evaluated on a held-out test set and saved to a file using pickle.

Dataset source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
"""

import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost.sklearn import XGBClassifier


# Load the data from a URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=column_names)

# Split the data into training and testing sets
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the XGBoost classifier with default hyperparameters
xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='logloss', seed=42)

# Define a grid of hyperparameters to search over
param_grid = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 5, 7, 8, 9, 10, 11],
    'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
}

# Perform grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print('Accuracy with best parameters:', accuracy_best)

# Save the best model to a file using pickle
filename = 'models/model_xgb.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)
