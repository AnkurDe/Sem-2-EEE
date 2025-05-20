import pandas as pd
import numpy as np
import shap
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

""" We have used two models in this code: CatBoostRegressor and ExtraTreesRegressor.
- CatBoostRegressor is used for predicting life expectancy.
- ExtraTreesRegressor is used for predicting health index.
Using two different models allows us to leverage the strengths of each algorithm for their respective tasks.
- We have used two models because we utilised the best of both worlds."""


# Path to input data file
DATA_FILE = 'Health index1.csv'

# Load the dataset into a pandas DataFrame
df = pd.read_csv(DATA_FILE)

# Data cleaning: Remove duplicate rows and rows with missing values
df = df.drop_duplicates().dropna()
print(f"Data shape after cleaning: {df.shape}")

# Remove outliers that are more than 3 standard deviations away from the mean
# This is done only for feature columns, not target variables
feature_cols = df.columns[:-2]  # Get all columns except the last 2 target columns
for col in feature_cols:
    mean, std = df[col].mean(), df[col].std()
    # Keep only rows where values are within ±3 std of mean
    df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]

print(f"Data shape after outlier removal: {df.shape}")

# Separate features (X) and target variables (y)
X = df.drop(columns=["Health index", "Life expectation"])  # Features
y = df[["Health index", "Life expectation"]]  # Target variables

# SHAP-based Feature Selection Process
# Step 1: Standardize features for SHAP analysis
scaler_shap = StandardScaler()
X_scaled = pd.DataFrame(scaler_shap.fit_transform(X), columns=X.columns)

# Step 2: Train a model for SHAP analysis using MultiOutputRegressor with ExtraTrees
model_shap = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=200, random_state=42))
model_shap.fit(X_scaled, y)

# Step 3: Calculate SHAP values using TreeExplainer
explainer = shap.TreeExplainer(model_shap.estimators_[0])
shap_values = explainer.shap_values(X_scaled)

# Step 4: Calculate feature importance based on mean absolute SHAP values
shap_means = np.abs(shap_values).mean(0)
shap_imports = pd.Series(shap_means, index=X.columns).sort_values(ascending=False)

# Display the top 8 most important features according to SHAP values
print("Top SHAP feature importance's:")
print(shap_imports.head(8))

# Select the top 8 most important features
top_feats = shap_imports.head(8).index.tolist()

# Create a dataset with only selected features
X_selected = X[top_feats]

# Apply rolling mean smoothing to reduce noise in both features and targets
to_smooth = top_feats + ['Health index', 'Life expectation']
df_smooth = df[to_smooth].rolling(window=5, min_periods=1).mean()

# Remove any NaN values that might have been introduced by smoothing
df_smooth = df_smooth.dropna()

print(f"Data shape after smoothing: {df_smooth.shape}")

# Prepare a final processed dataset
X_processed = df_smooth[top_feats]  # Features
y_processed = df_smooth[['Health index', 'Life expectation']]  # Targets

# Normalize all features and targets to have zero mean and unit variance
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_processed)

# IMPORTANT: Save the target normalization parameters before normalizing
target_means = y_processed.mean()
target_stds = y_processed.std()

# Now normalize the targets
y_norm = (y_processed - target_means) / target_stds

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

# Initialize CatBoost regressor for multi-target regression
cb = CatBoostRegressor(
    iterations=500,  # Number of boosting iterations
    learning_rate=0.1,  # Step size shrinkage
    depth=6,  # Maximum depth of trees
    loss_function='MultiRMSE',  # Loss function for multiple targets
    random_seed=42,
    verbose=0  # Suppress training output
)

# Train CatBoost model
cb.fit(X_train, y_train)

# Make predictions on a test set using CatBoost
y_pred_test = cb.predict(X_test)

# Calculate performance metrics for CatBoost
test_mse = mean_squared_error(y_test, y_pred_test, multioutput='raw_values')
test_r2 = r2_score(y_test, y_pred_test, multioutput='raw_values')

# Print CatBoost performance metrics
target_names = ['Health Index', 'Life Expectation']
print("CatBoost Testing Performance:\n")
print("catboost")
for i, name in enumerate(target_names):
    print(f"{name}:")
    print(f"  Test MSE: {test_mse[i]:.6f}")
    print(f"  Test R²: {test_r2[i]:.6f}")

# Train separate ExtraTrees models for each target variable
et_health = ExtraTreesRegressor(random_state=42).fit(X_train, y_train["Health index"])
et_life = ExtraTreesRegressor(random_state=42).fit(X_train, y_train["Life expectation"])

# Make predictions using ExtraTrees models
y1_pred_test = et_health.predict(X_test)  # Health index predictions
y2_pred_test = et_life.predict(X_test)  # Life expectation predictions

# Calculate and store performance metrics for ExtraTrees
errors_test = {
    "Health Index": {
        "MSE": mean_squared_error(y_test["Health index"], y1_pred_test),
        "R²": r2_score(y_test["Health index"], y1_pred_test)
    },
    "Life Expectation": {
        "MSE": mean_squared_error(y_test["Life expectation"], y2_pred_test),
        "R²": r2_score(y_test["Life expectation"], y2_pred_test)
    }
}

# Print ExtraTrees performance metrics
print("ExtraTreesRegressor Testing Errors:")
for target, metrics in errors_test.items():
    print(f"{target}: MSE = {metrics['MSE']:.6f}, R² = {metrics['R²']:.6f}")

# Create Hybrid Model: ExtraTrees for Health Index, CatBoost for Life Expectation
print("\n===== Hybrid Model Approach =====")
print("Using ExtraTrees for Health Index and CatBoost for Life Expectation")

# Get Life Expectation predictions from CatBoost
cb_life_pred = y_pred_test[:, 1]  # Extract second column (Life Expectation)

# Calculate and store hybrid model performance metrics
hybrid_metrics = {
    "Health Index (ExtraTrees)": {
        "MSE": errors_test["Health Index"]["MSE"],
        "R²": errors_test["Health Index"]["R²"]
    },
    "Life Expectation (CatBoost)": {
        "MSE": mean_squared_error(y_test["Life expectation"], cb_life_pred),
        "R²": r2_score(y_test["Life expectation"], cb_life_pred)
    }
}

# Print hybrid model performance metrics
print("\nHybrid Model Testing Performance:")
for target, metrics in hybrid_metrics.items():
    print(f"{target}:")
    print(f"  Test MSE: {metrics['MSE']:.6f}")
    print(f"  Test R²: {metrics['R²']:.6f}")

# Save the scalar constants for denormalization
scalar_constants = {
    'target_means': {
        'Health index': target_means['Health index'],
        'Life expectation': target_means['Life expectation']
    },
    'target_stds': {
        'Health index': target_stds['Health index'],
        'Life expectation': target_stds['Life expectation']
    }
}

# Save models and scalar constants
print("\nSaving models and scalar constants to files...")
joblib.dump(et_health, 'fl1.joblib')  # Save ExtraTrees model for Health Index
joblib.dump(cb, 'fl2.joblib')  # Save CatBoost model for Life Expectation
joblib.dump(scaler, 'scaler.joblib')  # Save the feature scaler
joblib.dump(scalar_constants, 'scalar_constants.joblib')  # Save scalar constants
print("Models and constants saved successfully")

# Print the saved constants for verification
print("\nSaved scalar constants:")
print(f"Health index - Mean: {scalar_constants['target_means']['Health index']:.6f}, Std: {scalar_constants['target_stds']['Health index']:.6f}")
print(f"Life expectation - Mean: {scalar_constants['target_means']['Life expectation']:.6f}, Std: {scalar_constants['target_stds']['Life expectation']:.6f}")