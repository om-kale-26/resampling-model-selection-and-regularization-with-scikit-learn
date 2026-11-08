"""
Resampling, Model Selection and Regularization with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import pandas as pd
from sklearn.datasets import load_diabetes

def load_data():
    # TODO: load_diabetes(as_frame=True); return (X DataFrame, y Series)
    data = load_diabetes(as_frame = True)
    X = data.data
    y = data.target
    return X, y

def describe_data(X, y):
    # TODO: {'n', 'p', 'features', 'y_mean' rounded to 2}
    summary = {
        'n' : X.shape[0],
        'p' : X.shape[1],
        'features': list(X.columns) ,
        'y_mean' : round(float(y.mean()), 2)
    }
    return summary

# Step 2 - train_test
from sklearn.model_selection import train_test_split

def train_test(X, y, test_size=0.25, random_state=0):
    # TODO: train_test_split; return (X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split (
        X, y, test_size = test_size, random_state = random_state
    )
    return X_train, X_test, y_train, y_test

# Step 3 - validation_set_curve
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def poly_model(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

def validation_set_curve(X, y, feature, degrees, random_state):
    X_feat = X[[feature]]
    X_train, X_val, y_train, y_val = train_test_split(
        X_feat, y, test_size=0.5, random_state=random_state
    )
    
    curve_results = {}
    for deg in degrees:
        model = poly_model(deg)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        curve_results[deg] = round(float(mse), 1)
        
    return curve_results

def curve_spread(X, y, feature, degrees, seeds):
    deg_errors = {deg: [] for deg in degrees}
    
    for seed in seeds:
        res = validation_set_curve(X, y, feature, degrees, seed)
        for deg, mse in res.items():
            deg_errors[deg].append(mse)
            
    spreads = {}
    for deg in degrees:
        spreads[deg] = round(float(max(deg_errors[deg]) - min(deg_errors[deg])), 1)
        
    return spreads

# Step 4 - cv_mse (not yet solved)
# TODO: implement

# Step 5 - cv_spread_by_k (not yet solved)
# TODO: implement

# Step 6 - bootstrap_coefficients (not yet solved)
# TODO: implement

# Step 7 - stepwise_path (not yet solved)
# TODO: implement

# Step 8 - score_path (not yet solved)
# TODO: implement

# Step 9 - one_se_rule (not yet solved)
# TODO: implement

# Step 10 - ridge_path (not yet solved)
# TODO: implement

# Step 11 - cv_curve (not yet solved)
# TODO: implement

# Step 12 - lasso_path (not yet solved)
# TODO: implement

# Step 13 - pcr_model (not yet solved)
# TODO: implement

# Step 14 - pls_model (not yet solved)
# TODO: implement

# Step 15 - fit_all (not yet solved)
# TODO: implement

# Step 16 - test_report (not yet solved)
# TODO: implement

