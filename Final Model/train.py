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

DATA_FILE = 'standardized_data.csv'

# Load data
df = pd.read_csv(DATA_FILE)

# Drop exact duplicates and any rows with missing values
df = df.drop_duplicates().dropna()
print(f"Data shape after cleaning: {df.shape}")

# Optional: remove obvious outliers (e.g., beyond 3σ in any feature)
# Only apply to feature columns, not targets
feature_cols = df.columns[:-2]  # Assuming last 2 are targets
for col in feature_cols:
    mean, std = df[col].mean(), df[col].std()
    df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]

print(f"Data shape after outlier removal: {df.shape}")

# Prepare initial data splits
X = df.drop(columns=["Health index", "Life expectation"])
y = df[["Health index", "Life expectation"]]

# SHAP-based feature selection
# First, standardize for SHAP analysis
scaler_shap = StandardScaler()
X_scaled = pd.DataFrame(scaler_shap.fit_transform(X), columns=X.columns)

# Train model for SHAP analysis
model_shap = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=200, random_state=42))
model_shap.fit(X_scaled, y)

# SHAP analysis - use first estimator for feature importance
explainer = shap.TreeExplainer(model_shap.estimators_[0])
shap_values = explainer.shap_values(X_scaled)

# Calculate feature importance based on mean absolute SHAP values
shap_means = np.abs(shap_values).mean(0)
shap_imports = pd.Series(shap_means, index=X.columns).sort_values(ascending=False)

print("Top SHAP feature importance's:")
print(shap_imports.head(8))

# Select top 8 features
top_feats = shap_imports.head(8).index.tolist()

# Apply feature selection
X_selected = X[top_feats]

# Apply smoothing to selected features and targets
to_smooth = top_feats + ['Health index', 'Life expectation']
df_smooth = df[to_smooth].rolling(window=5, min_periods=1).mean()

# Remove any remaining NaN values after smoothing
df_smooth = df_smooth.dropna()

print(f"Data shape after smoothing: {df_smooth.shape}")

# After all preprocessing, prepare final data
X_processed = df_smooth[top_feats]
y_processed = df_smooth[['Health index', 'Life expectation']]

# Normalize the data column-wise (final step)
X_norm = (X_processed - X_processed.mean()) / X_processed.std()
y_norm = (y_processed - y_processed.mean()) / y_processed.std()

# Split into train and test sets (final variables)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

from catboost import CatBoostRegressor

# Initialize CatBoost with MultiRMSE objective
cb = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiRMSE',
    random_seed=42,
    verbose=0
)

# Train the model
cb.fit(X_train, y_train)

# Predict on test data
y_pred_test = cb.predict(X_test)

# Calculate testing errors
test_mse = mean_squared_error(y_test, y_pred_test, multioutput='raw_values')
test_r2 = r2_score(y_test, y_pred_test, multioutput='raw_values')

# Print results
target_names = ['Health Index', 'Life Expectation']
print("CatBoost Testing Performance:\n")
print("catboost")
for i, name in enumerate(target_names):
    print(f"{name}:")
    print(f"  Test MSE: {test_mse[i]:.6f}")
    print(f"  Test R²: {test_r2[i]:.6f}")

# Initialize and train ExtraTrees models
et_health = ExtraTreesRegressor(random_state=42).fit(X_train, y_train["Health index"])
et_life = ExtraTreesRegressor(random_state=42).fit(X_train, y_train["Life expectation"])

# Predict on test data
y1_pred_test = et_health.predict(X_test)
y2_pred_test = et_life.predict(X_test)

# Calculate testing errors
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
# Print results
print("ExtraTreesRegressor Testing Errors:")
for target, metrics in errors_test.items():
    print(f"{target}: MSE = {metrics['MSE']:.6f}, R² = {metrics['R²']:.6f}")

# ===== Hybrid Model Approach =====
# Using CatBoost for Life Expectation and ExtraTrees for Health Index
print("\n===== Hybrid Model Approach =====")
print("Using ExtraTrees for Health Index and CatBoost for Life Expectation")

# Extract the Life Expectation prediction from CatBoost
cb_life_pred = y_pred_test[:, 1]  # Second column is Life Expectation

# Calculate metrics for the hybrid model
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

# Print hybrid results
print("\nHybrid Model Testing Performance:")
for target, metrics in hybrid_metrics.items():
    print(f"{target}:")
    print(f"  Test MSE: {metrics['MSE']:.6f}")
    print(f"  Test R²: {metrics['R²']:.6f}")

# Save the models to joblib files
print("\nSaving models to files...")
joblib.dump(et_health, 'fl1.joblib')  # Health Index model (ExtraTrees)
joblib.dump(cb, 'fl2.joblib')         # Life Expectation model (CatBoost)
print("Models saved to fl1.joblib and fl2.joblib")