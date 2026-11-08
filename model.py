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

# Step 8 - score_path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def score_path(X, y, path, cv):
    sizes = []
    means = []
    ses = []
    
    n_splits = cv.get_n_splits(X) if hasattr(cv, "get_n_splits") else 5
    
    for k, features in path.items():
        X_subset = X[features]
        # Get negative MSE scores
        scores = cross_val_score(
            LinearRegression(), X_subset, y, cv=cv, scoring='neg_mean_squared_error'
        )
        # Convert to positive MSE
        mse_scores = -scores
        
        mean_val = float(np.mean(mse_scores))
        se_val = float(np.std(mse_scores, ddof=1) / np.sqrt(n_splits))
        
        sizes.append(k)
        means.append(round(mean_val, 1))
        ses.append(round(se_val, 1))
        
    return sizes, means, ses

def best_size(sizes, means):
    # Find the index of the minimum mean error
    min_idx = int(np.argmin(means))
    return sizes[min_idx]

# Step 9 - one_se_rule
import numpy as np

def one_se_rule(values, means, ses, prefer='smaller'):
    values = np.array(values)
    means = np.array(means)
    ses = np.array(ses)
    
    # 1. Locate the model with the minimum error
    min_idx = np.argmin(means)
    
    # 2. Define the threshold: minimum error + standard error of that specific minimum model
    threshold = means[min_idx] + ses[min_idx]
    
    # 3. Filter models whose mean is at most the threshold
    valid_mask = means <= threshold
    valid_values = values[valid_mask]
    
    if prefer == 'smaller':
        return valid_values[0]  # Simplest model within 1 SE
    else:
        return valid_values[-1] # Largest model within 1 SE

def choose_subset(X, y, direction, cv):
    # 1. Run the stepwise path
    path = stepwise_path(X, y, direction, cv)
    
    # 2. Score the path using cross-validation
    sizes, means, ses = score_path(X, y, path, cv)
    
    # 3. Find the best size using the minimum mean error
    size_min = best_size(sizes, means)
    
    # 4. Find the size using the one-standard-error rule
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    
    # 5. Get the features corresponding to size_1se
    # Check if path is a dict/mapping or list, and retrieve features correctly matching size_1se
    if isinstance(path, dict):
        features_1se = path[size_1se]
    else:
        # Find the exact index where sizes equals size_1se
        try:
            idx = list(sizes).index(size_1se)
            features_1se = path[idx]
        except (ValueError, IndexError):
            features_1se = path[size_1se]
    
    return size_min, size_1se, features_1se

# Step 10 - ridge_path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def ridge_model(alpha):
    # Create pipeline with StandardScaler and Ridge regression
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    coefs = []
    for alpha in alphas:
        model = ridge_model(alpha)
        model.fit(X, y)
        # Extract the coefficients from the Ridge step of the pipeline
        coefs.append(model.named_steps['ridge'].coef_)
    return np.array(coefs)

def coef_norms(path):
    # Compute the L2 norm for each row and round to 2 decimals
    norms = np.linalg.norm(path, axis=1)
    return np.round(norms, 2)

# Step 11 - cv_curve
import numpy as np
from sklearn.model_selection import cross_val_score

def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []
    
    for val in values:
        model = make_model(val)
        # Get negative mean squared errors from cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        
        # Convert negative MSE to positive MSE error for evaluation
        mse_scores = -scores
        
        mean_err = np.mean(mse_scores)
        # Standard error: standard deviation divided by square root of number of folds
        se_err = np.std(mse_scores, ddof=1) / np.sqrt(len(mse_scores))
        
        means.append(round(float(mean_err), 1))
        ses.append(round(float(se_err), 1))
        
    return means, ses

def choose_penalty(make_model, X, y, values, cv):
    # Compute the cross-validation curve
    means, ses = cv_curve(make_model, X, y, values, cv)
    
    # Find the value that gives the minimum mean error
    min_idx = np.argmin(means)
    value_min = values[min_idx]
    
    # Use the one-SE rule preferring larger penalty values (more regularization)
    value_1se = one_se_rule(values, means, ses, prefer='larger')
    
    return value_min, value_1se

# Step 12 - lasso_path
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def lasso_model(alpha):
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))

def lasso_path(X, y, alphas):
    coefs = []
    for alpha in alphas:
        model = lasso_model(alpha)
        model.fit(X, y)
        coefs.append(model.named_steps['lasso'].coef_)
    return np.array(coefs)

def nonzero_features(coef, names, tol=1e-8):
    coef = np.array(coef)
    names = np.array(names)
    # Filter and convert back to a standard list of Python strings
    mask = np.abs(coef) > tol
    return names[mask].tolist()

def lasso_cv(X, y, cv):
    model = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0))
    model.fit(X, y)
    
    lasso_cv_step = model.named_steps['lassocv']
    alpha_val = round(float(lasso_cv_step.alpha_), 4)
    coefs = lasso_cv_step.coef_
    
    if hasattr(X, "columns"):
        names = list(X.columns)
    else:
        names = [f"x{i}" for i in range(X.shape[1])]
        
    selected = nonzero_features(coefs, names)
    return alpha_val, selected

# Step 13 - pcr_model
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def pcr_model(n_components):
    return make_pipeline(
        StandardScaler(), 
        PCA(n_components=n_components), 
        LinearRegression()
    )

def explained_variance(X):
    # Fit StandardScaler then PCA on X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calculate cumulative explained variance ratio and round to 3 decimals
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    return np.round(cum_var, 3)

def pcr_curve(X, y, cv):
    p = X.shape[1]
    components = list(range(1, p + 1))
    
    # Use cv_curve from earlier steps to evaluate the PCR models for each component count
    means, ses = cv_curve(pcr_model, X, y, components, cv)
    
    return components, means, ses

# Step 14 - pls_model
import numpy as np
from sklearn.cross_decomposition import PLSRegression

def pls_model(n_components):
    return PLSRegression(n_components=n_components, scale=True)

def pls_curve(X, y, cv):
    p = X.shape[1]
    components = list(range(1, p + 1))
    means, ses = cv_curve(pls_model, X, y, components, cv)
    return components, means, ses

def pls_predict(model, X):
    # PLSRegression predicts a 2D column matrix, flatten it to a 1D array
    preds = model.predict(X)
    return np.ravel(preds)

def best_components(components, means, ses):
    min_idx = np.argmin(means)
    m_min = components[min_idx]
    m_1se = one_se_rule(components, means, ses, prefer='smaller')
    return m_min, m_1se

# Step 15 - fit_all
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def fit_all(X, y, cv, alphas):
    # Get all feature names if X is a DataFrame, or indices
    if hasattr(X, "columns"):
        all_features = list(X.columns)
    else:
        all_features = list(range(X.shape[1]))
        
    models = {}
    
    # 1. OLS
    ols_model = LinearRegression()
    ols_model.fit(X, y)
    models['ols'] = (ols_model, all_features, None)
    
    # 2. Forward Selection (one-SE rule)
    _, size_1se, feats_1se = choose_subset(X, y, 'forward', cv)
    forward_model = LinearRegression()
    forward_model.fit(X[feats_1se], y)
    models['forward_1se'] = (forward_model, feats_1se, size_1se)
    
    # 3. Ridge (one-SE penalty)
    _, alpha_1se = choose_penalty(ridge_model, X, y, alphas, cv)
    ridge_m = ridge_model(alpha_1se)
    ridge_m.fit(X, y)
    models['ridge_1se'] = (ridge_m, all_features, alpha_1se)
    
    # 4. Lasso CV
    lasso_cv_pipe = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0))
    lasso_cv_pipe.fit(X, y)
    best_alpha = round(float(lasso_cv_pipe.named_steps['lassocv'].alpha_), 4)
    models['lasso_cv'] = (lasso_cv_pipe, all_features, best_alpha)
    
    # 5. PCR (one-SE components)
    pcr_comps, pcr_means, pcr_ses = pcr_curve(X, y, cv)
    pcr_min_comp, pcr_1se_comp = best_components(pcr_comps, pcr_means, pcr_ses)
    pcr_m = pcr_model(pcr_1se_comp)
    pcr_m.fit(X, y)
    models['pcr_1se'] = (pcr_m, all_features, pcr_1se_comp)
    
    # 6. PLS (one-SE components)
    pls_comps, pls_means, pls_ses = pls_curve(X, y, cv)
    pls_min_comp, pls_1se_comp = best_components(pls_comps, pls_means, pls_ses)
    pls_m = pls_model(pls_1se_comp)
    pls_m.fit(X, y)
    models['pls_1se'] = (pls_m, all_features, pls_1se_comp)
    
    return models

# Step 16 - test_report (not yet solved)
# TODO: implement

