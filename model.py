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

# Step 2 - train_test (not yet solved)
# TODO: implement

# Step 3 - validation_set_curve (not yet solved)
# TODO: implement

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

