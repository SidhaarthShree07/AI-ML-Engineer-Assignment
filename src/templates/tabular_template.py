"""Tabular data strategy template with LightAutoML.

Sophisticated template for tabular datasets that includes:
- Automatic feature engineering (string character extraction, interactions)
- Proper preprocessing with StandardScaler
- LightAutoML for model selection
- Cross-validation with configurable folds
- LLM customizes column names and task-specific logic
"""

TABULAR_TEMPLATE = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - LLM MUST CUSTOMIZE BASED ON COMPETITION DATA
# ============================================================================
# Look at train_data_preview and test_data_preview to set these correctly

ID_COLUMN = '{id_column}'           # Column for submission IDs
TARGET_COLUMN = '{target_column}'   # Target variable (only in train)
PREDICTION_COLUMN = '{prediction_column}'  # Column name for submission predictions

# Task configuration (LLM should set based on competition description)
TASK_TYPE = 'binary'  # 'binary', 'multiclass', or 'reg'

# Training configuration
SEED = {seed}
N_FOLDS = {n_folds}
TIMEOUT = {time_budget}  # seconds for AutoML
THREADS = 12  # CPU threads

print("=" * 60)
print("TABULAR PIPELINE - LightAutoML")
print("=" * 60)

# ============================================================================
# LOAD DATA
# ============================================================================

train_path = '{train_path}'
test_path = '{test_path}'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(f"Train shape: {{df_train.shape}}")
print(f"Test shape: {{df_test.shape}}")
print(f"Train columns: {{list(df_train.columns)}}")
print(f"Test columns: {{list(df_test.columns)}}")

# Save test IDs for submission
if ID_COLUMN in df_test.columns:
    test_ids = df_test[ID_COLUMN].copy()
else:
    test_ids = pd.Series(range(len(df_test)), name=ID_COLUMN)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):
    \"\"\"Generic feature engineering for tabular data.\"\"\"
    new_df = df.copy()
    
    # Find string columns (excluding ID and target)
    string_cols = new_df.select_dtypes(include=['object']).columns.tolist()
    string_cols = [c for c in string_cols if c not in [ID_COLUMN, TARGET_COLUMN]]
    
    for col in string_cols:
        # Extract character features from string columns
        max_len = new_df[col].astype(str).str.len().max()
        for i in range(min(int(max_len), 10)):
            new_df[f'{{{{col}}}}_ch{{{{i}}}}'] = new_df[col].astype(str).str.get(i).apply(
                lambda x: ord(x) - ord('A') if pd.notna(x) and x.isalpha() else -1
            )
        # Count unique characters
        new_df[f'{{{{col}}}}_unique_chars'] = new_df[col].astype(str).apply(lambda s: len(set(s)) if pd.notna(s) else 0)
    
    return new_df

# Apply feature engineering
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)

print(f"Features after engineering - Train: {{df_train.shape}}, Test: {{df_test.shape}}")

# ============================================================================
# PREPROCESSING
# ============================================================================

# Get feature columns (exclude ID, target, and original string columns)
string_cols = df_train.select_dtypes(include=['object']).columns.tolist()
features = [f for f in df_train.columns if f not in [ID_COLUMN, TARGET_COLUMN] + string_cols]

print(f"Number of features: {{len(features)}}")

# Scale features
scaler = StandardScaler()
df_train[features] = scaler.fit_transform(df_train[features])
df_test[features] = scaler.transform(df_test[features])

# Drop ID and string columns
drop_cols = [ID_COLUMN] + string_cols
df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')

# ============================================================================
# MODEL TRAINING WITH LightAutoML
# ============================================================================

print("\\nTraining with LightAutoML...")

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

# Setup task
task = Task(TASK_TYPE)
roles = {{'target': TARGET_COLUMN}}

# Create and train AutoML
automl = TabularAutoML(
    task=task,
    timeout=TIMEOUT,
    cpu_limit=THREADS,
    reader_params={{'n_jobs': THREADS, 'cv': N_FOLDS, 'random_state': SEED}}
)

# Fit and get OOF predictions
oof_pred = automl.fit_predict(df_train, roles=roles, verbose=1)

print("LightAutoML training complete!")

# ============================================================================
# PREDICTION
# ============================================================================

print("\\nMaking predictions on test set...")

# Get test predictions
test_pred = automl.predict(df_test)

# Extract predictions
if TASK_TYPE == 'binary':
    predictions = test_pred.data[:, 0]
elif TASK_TYPE == 'multiclass':
    predictions = test_pred.data.argmax(axis=1) if test_pred.data.ndim > 1 else test_pred.data
else:
    predictions = test_pred.data[:, 0] if test_pred.data.ndim > 1 else test_pred.data

# ============================================================================
# CREATE SUBMISSION
# ============================================================================

print("\\nCreating submission...")

submission = pd.DataFrame()
submission[ID_COLUMN] = test_ids
submission[PREDICTION_COLUMN] = predictions

submission.to_csv('submission.csv', index=False)
print(f"Submission saved! Shape: {{submission.shape}}")
print(submission.head())

print("\\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
"""

# Keep aliases for backward compatibility
TABULAR_SMALL_TEMPLATE = TABULAR_TEMPLATE
TABULAR_LARGE_TEMPLATE = TABULAR_TEMPLATE
TABULAR_RESOURCE_CONSTRAINED_TEMPLATE = TABULAR_TEMPLATE


def get_tabular_template(memory_gb: float = 0, resource_constrained: bool = False, num_samples: int = 0) -> str:
    """
    Return the tabular template.
    
    Args:
        memory_gb: Ignored (kept for compatibility)
        resource_constrained: Ignored (kept for compatibility)
        num_samples: Ignored (kept for compatibility)
        
    Returns:
        Template string with placeholders
    """
    return TABULAR_TEMPLATE
