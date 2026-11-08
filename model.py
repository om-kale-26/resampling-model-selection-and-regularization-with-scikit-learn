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

# Step 4 - cv_mse
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

def cv_mse(estimator, X, y, k=5, random_state=0):
    # Set up KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # Compute negative mean squared error scores across folds
    scores = cross_val_score(estimator, X, y, cv=kf, scoring='neg_mean_squared_error')
    
    # Convert to positive MSE
    mse_scores = -scores
    
    mean_val = float(np.mean(mse_scores))
    
    # Standard error: standard deviation (ddof=1) divided by sqrt(k)
    se_val = float(np.std(mse_scores, ddof=1) / np.sqrt(k))
    
    return round(mean_val, 2), round(se_val, 2)

def loocv_mse(estimator, X, y):
    # Set up Leave-One-Out cross-validation
    loo = LeaveOneOut()
    
    # Compute negative mean squared error scores
    scores = cross_val_score(estimator, X, y, cv=loo, scoring='neg_mean_squared_error')
    
    # Convert to positive MSE and get the average over all held-out points
    mse_scores = -scores
    mean_val = float(np.mean(mse_scores))
    
    return round(mean_val, 2)

# Step 5 - cv_spread_by_k
import numpy as np

def cv_spread_by_k(estimator, X, y, ks, seeds):
    results = {}
    for k in ks:
        seed_means = []
        for seed in seeds:
            # Call cv_mse from previous step; index [0] is the mean MSE
            mean_mse, _ = cv_mse(estimator, X, y, k=k, random_state=seed)
            seed_means.append(mean_mse)
        
        avg_over_seeds = float(np.mean(seed_means))
        std_over_seeds = float(np.std(seed_means, ddof=1))
        
        results[k] = (round(avg_over_seeds, 1), round(std_over_seeds, 1))
        
    return results

def compare_with_loocv(estimator, X, y, ks, seeds):
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator, X, y, ks, seeds)
    }

# Step 6 - bootstrap_coefficients
import numpy as np
from sklearn.linear_model import LinearRegression

def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(X)
    coefs = []
    
    for _ in range(n_boot):
        # Sample row indices with replacement
        indices = rng.integers(0, n, n)
        X_b = X.iloc[indices]
        y_b = y.iloc[indices]
        
        model = LinearRegression().fit(X_b, y_b)
        coefs.append(model.coef_)
        
    return np.array(coefs)

def bootstrap_se(coefs):
    # Standard deviation over bootstrap replicates (ddof=1), rounded to 2 decimals
    se = np.std(coefs, axis=0, ddof=1)
    return np.round(se, 2)

def ols_standard_errors(X, y):
    # Convert inputs to numpy arrays if they are pandas
    X_arr = X.values if hasattr(X, "values") else np.array(X)
    y_arr = y.values if hasattr(y, "values") else np.array(y)
    
    n, p = X_arr.shape
    
    # Design matrix A = [1, X]
    A = np.hstack([np.ones((n, 1)), X_arr])
    
    # Fit model to get predictions and residuals for RSS
    model = LinearRegression().fit(X_arr, y_arr)
    preds = model.predict(X_arr)
    rss = np.sum((y_arr - preds) ** 2)
    
    # Variance estimate sigma^2 = RSS / (n - p - 1)
    sigma2 = rss / (n - p - 1)
    
    # Standard errors: sqrt(sigma^2 * diag((A^T A)^-1))
    inv_ATA = np.linalg.inv(A.T @ A)
    se_full = np.sqrt(sigma2 * np.diagonal(inv_ATA))
    
    # Drop the intercept entry (first element) and round to 2 decimals
    return np.round(se_full[1:], 2)

# Step 7 - stepwise_path
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

def select_features(X, y, k, direction, cv):
    # TODO: SequentialFeatureSelector with neg MSE scoring; return the selected column names
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select = k,
        direction = direction,
        cv = cv,
        scoring = 'neg_mean_squared_error'
    )
    sfs.fit(X , y)

    selected_mask = sfs.get_support()
    return list(X.columns[selected_mask])



def stepwise_path(X, y, direction, cv):
    # TODO: {k: selected features} for k = 1..p-1, plus k = p -> all columns
    p = X.shape[1]
    path = {}
    
    # Loop for k from 1 up to p-1, plus p -> all columns
    for k in range(1, p + 1):
        if k == p:
            path[k] = list(X.columns)
        else:
            path[k] = select_features(X, y, k, direction, cv)
            
    return path

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

