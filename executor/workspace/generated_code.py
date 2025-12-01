import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import VotingClassifier, VotingRegressor
import pickle
import os
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# AUTO FEATURE ENGINEERING FUNCTIONS
# These detect patterns and create features automatically without hardcoding
# =============================================================================

def auto_feature_engineer(df, is_train=True, feature_stats=None):
    """
    Automatically detect and engineer features from the dataset.
    - Parses string columns into character-level features
    - Creates interaction features based on correlation patterns
    - Adds unique character counts for string columns
    
    Args:
        df: DataFrame to engineer
        is_train: Whether this is training data (to compute statistics)
        feature_stats: Pre-computed stats from training (for test data)
    
    Returns:
        Engineered DataFrame and feature statistics
    """
    df = df.copy()
    stats = feature_stats or {}
    
    # 1. DETECT AND PARSE STRING COLUMNS
    # Find string columns that have fixed-length alphanumeric patterns (like f_27)
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in string_cols:
        if col in ['id', 'target']:
            continue
            
        # Check if column has consistent string length (suggests parseable pattern)
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue
            
        lengths = sample.astype(str).str.len()
        if lengths.std() < 1 and lengths.mean() >= 3:  # Fixed-length strings
            str_len = int(lengths.mode().iloc[0]) if len(lengths.mode()) > 0 else int(lengths.mean())
            
            # Create character position features (ordinal encoded)
            for i in range(min(str_len, 10)):  # Max 10 character positions
                new_col = f'{col}_char_{i}'
                df[new_col] = df[col].astype(str).str[i].apply(
                    lambda x: ord(x.upper()) - ord('A') if pd.notna(x) and x.isalpha() else -1
                )
            
            # Create unique characters feature
            df[f'{col}_unique_chars'] = df[col].astype(str).apply(lambda x: len(set(x)))
            
            # Drop original string column
            df = df.drop(columns=[col])
    
    # 2. CREATE INTERACTION FEATURES FOR NUMERIC COLUMNS
    # Detect highly correlated feature pairs and create threshold-based interactions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['id', 'target']]
    
    if is_train and len(numeric_cols) > 2:
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs (|r| > 0.3)
        interaction_pairs = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if corr_matrix.loc[col1, col2] > 0.3:
                    interaction_pairs.append((col1, col2))
        
        # Create interaction features for top pairs
        stats['interaction_pairs'] = interaction_pairs[:5]  # Max 5 interaction pairs
        
        for col1, col2 in stats.get('interaction_pairs', []):
            if col1 in df.columns and col2 in df.columns:
                # Sum interaction
                sum_col = f'i_{col1}_{col2}_sum'
                df[sum_col] = df[col1] + df[col2]
                
                # Threshold-based interaction (like i_02_21 in reference)
                # Find natural thresholds using percentiles
                combined = df[col1] + df[col2]
                if is_train:
                    upper_thresh = combined.quantile(0.85)
                    lower_thresh = combined.quantile(0.15)
                    stats[f'{sum_col}_upper'] = upper_thresh
                    stats[f'{sum_col}_lower'] = lower_thresh
                else:
                    upper_thresh = stats.get(f'{sum_col}_upper', combined.quantile(0.85))
                    lower_thresh = stats.get(f'{sum_col}_lower', combined.quantile(0.15))
                
                thresh_col = f'i_{col1}_{col2}_thresh'
                df[thresh_col] = (combined > upper_thresh).astype(int) - (combined < lower_thresh).astype(int)
    else:
        # Apply pre-computed interactions for test data
        for col1, col2 in stats.get('interaction_pairs', []):
            if col1 in df.columns and col2 in df.columns:
                sum_col = f'i_{col1}_{col2}_sum'
                df[sum_col] = df[col1] + df[col2]
                
                combined = df[col1] + df[col2]
                upper_thresh = stats.get(f'{sum_col}_upper', combined.quantile(0.85))
                lower_thresh = stats.get(f'{sum_col}_lower', combined.quantile(0.15))
                
                thresh_col = f'i_{col1}_{col2}_thresh'
                df[thresh_col] = (combined > upper_thresh).astype(int) - (combined < lower_thresh).astype(int)
    
    return df, stats


def rank_based_ensemble_predict(models, X_test, is_classification=True):
    """
    Create rank-based ensemble predictions for ROC AUC optimization.
    Instead of averaging probabilities, average ranks for better calibration.
    
    Args:
        models: List of trained models
        X_test: Test features
        is_classification: Whether this is a classification task
    
    Returns:
        Ensemble predictions
    """
    if not is_classification:
        # For regression, just average predictions
        return np.mean([m.predict(X_test) for m in models], axis=0)
    
    pred_list = []
    for model in models:
        try:
            preds = model.predict_proba(X_test)[:, 1]
        except:
            preds = model.predict(X_test)
        pred_list.append(preds)
    
    # Convert to ranks and average
    pred_df = pd.DataFrame(pred_list).T
    rank_df = pred_df.rank()
    return rank_df.mean(axis=1).values


# =============================================================================
# MAIN TRAINING LOGIC
# =============================================================================

# Load data
try:
    train_df = pd.read_csv('data_test/train/train.csv')
    test_df = pd.read_csv('data_test/test/test.csv')
    logging.info("Data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Save test IDs before any processing
if 'id' in test_df.columns:
    test_ids = test_df['id'].copy()
else:
    test_ids = pd.Series(range(len(test_df)), name='id')

# Apply auto feature engineering
print("Applying auto feature engineering...")
train_df, feature_stats = auto_feature_engineer(train_df, is_train=True)
test_df, _ = auto_feature_engineer(test_df, is_train=False, feature_stats=feature_stats)

print(f"After feature engineering - Train: {train_df.shape}, Test: {test_df.shape}")

# Separate features and target
X = train_df.drop(columns=['target'], errors='ignore')
y = train_df['target']

# Drop ID column from both train and test features
if 'id' in X.columns:
    X = X.drop(columns=['id'])

if 'id' in test_df.columns:
    X_test = test_df.drop(columns=['id'])
else:
    X_test = test_df.copy()

# Remove target column from test if present
if 'target' in X_test.columns:
    X_test = X_test.drop(columns=['target'])

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Handle missing values for numeric columns
if numeric_cols:
    imputer_numeric = SimpleImputer(strategy='median')
    X[numeric_cols] = imputer_numeric.fit_transform(X[numeric_cols])
    X_test[numeric_cols] = imputer_numeric.transform(X_test[numeric_cols])

# Handle missing values and encode categorical columns
if categorical_cols:
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = imputer_categorical.fit_transform(X[categorical_cols])
    X_test[categorical_cols] = imputer_categorical.transform(X_test[categorical_cols])
    
    # Encode categorical features with handling for unseen labels
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
        # Handle unseen labels in test set
        test_col_values = X_test[col].astype(str)
        test_encoded = []
        for val in test_col_values:
            if val in le.classes_:
                test_encoded.append(le.transform([val])[0])
            else:
                test_encoded.append(-1)
        X_test[col] = test_encoded

X_imputed = X
X_test_imputed = X_test

# Ensure all column names are strings (required for sklearn)
X_imputed.columns = X_imputed.columns.astype(str)
X_test_imputed.columns = X_test_imputed.columns.astype(str)

# Add polynomial features (degree 2)
try:
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_imputed)
    X_test_poly = poly.transform(X_test_imputed)

    # Convert to pandas DataFrame
    X_poly = pd.DataFrame(X_poly, columns=[f'poly_{i}' for i in range(X_poly.shape[1])])
    X_test_poly = pd.DataFrame(X_test_poly, columns=[f'poly_{i}' for i in range(X_test_poly.shape[1])])
    
    # Concatenate polynomial features with original features
    X_imputed = pd.concat([X_imputed.reset_index(drop=True), X_poly.reset_index(drop=True)], axis=1)
    X_test_imputed = pd.concat([X_test_imputed.reset_index(drop=True), X_test_poly.reset_index(drop=True)], axis=1)
    
    logging.info("Polynomial features added successfully.")

except Exception as e:
    logging.warning(f"Polynomial feature generation failed: {e}")


# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Detect task type automatically
n_unique = y.nunique()
is_classification = n_unique < 20 and y.dtype in ['object', 'int64', 'int32']

if is_classification:
    task_type = 'classification'
    if n_unique == 2:
        metric = 'roc_auc'
    else:
        metric = 'log_loss'
else:
    task_type = 'regression'
    metric = 'r2'

print(f"Detected task type: {task_type}")
print(f"Using metric: {metric}")

# FLAML AutoML with GPU support
automl = AutoML()
automl_settings = {
    "time_budget": 51840,
    "metric": metric,
    "task": task_type,
    "log_file_name": 'flaml_log.txt',
    "seed": 42,
    "early_stop": True,
    "verbose": 3,
    "use_ray": False,
    "n_jobs": -1,
    "estimator_list": ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree'],
    "ensemble": True,
    "eval_method": "cv",
    "split_type": "stratified",
    "n_splits": 5,
}

# Train with FLAML
automl.fit(X_scaled, y, **automl_settings)

# Make predictions
if is_classification and n_unique == 2:
    # For binary classification, get probabilities
    predictions = automl.predict_proba(X_test_scaled)[:, 1]
elif is_classification:
    # For multi-class, get class predictions
    predictions = automl.predict(X_test_scaled)
else:
    # For regression, get continuous predictions
    predictions = automl.predict(X_test_scaled)

# Save predictions
submission = pd.DataFrame({
    'id': test_ids,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)

print(f"Best model: {automl.best_estimator}")
print(f"Best validation score: {automl.best_loss}")
print("Training complete!")