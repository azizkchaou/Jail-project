"""
Random Forest Model with One-Hot Encoding to Predict JAIL_DAYS

This script:
1. Loads the preprocessed jail data (jail_filtred.csv)
2. Applies one-hot encoding to categorical features
3. Trains a Random Forest regression model to predict JAIL_DAYS
4. Evaluates the model with metrics (MAE, RMSE, R²)
5. Visualizes feature importance
6. Saves predictions and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Random Forest Model with Label Encoding: Predicting JAIL_DAYS")
print("=" * 80)

# --- Load preprocessed dataset ---
df = pd.read_csv(r"C:\Users\USER\Desktop\jail project\jail_filtred.csv", 
                 encoding='utf-8', 
                 low_memory=False)
print("Loaded jail_filtred.csv")
print(f"Dataset shape: {df.shape}")
print(f"JAIL_DAYS missing: {df['JAIL_DAYS'].isna().sum()}")

# --- Prepare features and target ---
df_model = df.dropna(subset=['JAIL_DAYS']).copy()
print(f"Rows with valid JAIL_DAYS: {len(df_model)}")

y = df_model['JAIL_DAYS'].values

# Select features (exclude COMMITMENT_TERM, COMMITMENT_UNIT, JAIL_DAYS)
exclude_cols = {'JAIL_DAYS', 'COMMITMENT_TERM', 'COMMITMENT_UNIT'}
numeric_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in exclude_cols]

print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols[:5]}..." if len(numeric_cols) > 5 else f"\nNumeric features: {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols[:5]}..." if len(categorical_cols) > 5 else f"Categorical features: {categorical_cols}")

# Prepare feature matrix
X = df_model[numeric_cols + categorical_cols].copy()

# Handle missing values
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
X[categorical_cols] = X[categorical_cols].fillna('Unknown')

print(f"\nFeature matrix shape (before encoding): {X.shape}")

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# --- One-Hot Encoding ---
print("\n" + "=" * 80)
print("Applying Label Encoding to categorical features...")
print("=" * 80)

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined train+test data to ensure all categories are covered
    le.fit(pd.concat([X_train[col], X_test[col]]))
    X_train_encoded[col] = le.transform(X_train[col])
    X_test_encoded[col] = le.transform(X_test[col])
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} unique categories encoded")

print(f"\nFeature matrix shape (after encoding): {X_train_encoded.shape}")
print(f"Training set features: {X_train_encoded.shape[1]}")
print(f"Test set features: {X_test_encoded.shape[1]}")

# --- Train Random Forest Model ---
print("\n" + "=" * 80)
print("Training Random Forest Model...")
print("=" * 80)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train_encoded, y_train)
print("Model training complete!")

# --- Evaluate Model ---
print("\n" + "=" * 80)
print("Model Evaluation")
print("=" * 80)

y_pred_train = model.predict(X_train_encoded)
y_pred_test = model.predict(X_test_encoded)

mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"\nTraining Metrics:")
print(f"  MAE:  {mae_train:.2f} days")
print(f"  RMSE: {rmse_train:.2f} days")
print(f"  R²:   {r2_train:.4f}")

print(f"\nTest Metrics:")
print(f"  MAE:  {mae_test:.2f} days")
print(f"  RMSE: {rmse_test:.2f} days")
print(f"  R²:   {r2_test:.4f}")


# Compute Accuracy and Precision (regression to classification via tolerance)
# Define accuracy as predictions within �15% of actual values
tolerance = 0.15
y_pred_train_binary = (np.abs(y_pred_train - y_train) <= y_train * tolerance).astype(int)
y_pred_test_binary = (np.abs(y_pred_test - y_test) <= y_test * tolerance).astype(int)

accuracy_train = np.mean(y_pred_train_binary)  # Proportion of "correct" predictions
accuracy_test = np.mean(y_pred_test_binary)

# Precision: among predictions considered "correct" (within tolerance), what fraction are truly good?
# For regression, we'll use: correct predictions / total predictions
precision_train = accuracy_train  # Same as accuracy for this threshold approach
precision_test = accuracy_test


# Compute Accuracy and Precision (regression to classification via tolerance)
# Define accuracy as predictions within �15% of actual values
tolerance = 0.15
y_pred_train_binary = (np.abs(y_pred_train - y_train) <= y_train * tolerance).astype(int)
y_pred_test_binary = (np.abs(y_pred_test - y_test) <= y_test * tolerance).astype(int)

accuracy_train = np.mean(y_pred_train_binary)
accuracy_test = np.mean(y_pred_test_binary)
precision_train = accuracy_train
precision_test = accuracy_test

# --- Feature Importance ---
print("\n" + "=" * 80)
print("Top 15 Most Important Features")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': X_train_encoded.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# --- Save Results ---
results_dir = 'catboost_outputs'
import os
os.makedirs(results_dir, exist_ok=True)

# Save model predictions
predictions_df = pd.DataFrame({
    'Actual_JAIL_DAYS': y_test,
    'Predicted_JAIL_DAYS': y_pred_test,
    'Error': y_test - y_pred_test,
    'Absolute_Error': np.abs(y_test - y_pred_test)
})
predictions_df.to_csv(os.path.join(results_dir, 'test_predictions.csv'), index=False)
print(f"\nTest predictions saved to {results_dir}/test_predictions.csv")

# Save feature importance
feature_importance.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
print(f"Feature importance saved to {results_dir}/feature_importance.csv")

# --- Visualizations ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted (Test Set)
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual JAIL_DAYS')
axes[0, 0].set_ylabel('Predicted JAIL_DAYS')
axes[0, 0].set_title('Actual vs Predicted (Test Set)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals
residuals = y_test - y_pred_test
axes[0, 1].scatter(y_pred_test, residuals, alpha=0.5, s=20)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted JAIL_DAYS')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot (Test Set)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Distribution of Residuals
axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Residuals')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Top 10 Feature Importance
top_features = feature_importance.head(10)
axes[1, 1].barh(top_features['Feature'], top_features['Importance'], color='steelblue')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Top 10 Feature Importance')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_evaluation.png'), dpi=100, bbox_inches='tight')
print(f"Evaluation plots saved to {results_dir}/model_evaluation.png")
# For the tolerance-based binary classification, define ground-truth as all positives
# (each actual value is considered a 'positive' for being within tolerance); recall
# will equal accuracy here, but we compute it explicitly for clarity.
y_true_train_binary = np.ones_like(y_pred_train_binary)
y_true_test_binary = np.ones_like(y_pred_test_binary)
recall_train = recall_score(y_true_train_binary, y_pred_train_binary, zero_division=0)
recall_test = recall_score(y_true_test_binary, y_pred_test_binary, zero_division=0)

# Print Accuracy and Precision 
print("\n" + "=" * 20 + " Final Accuracy/Precision " + "=" * 20)
print(f"Tolerance used: {tolerance*100:.0f}% of actual value")
print(f"Train Accuracy:   {accuracy_train:.4f}")
print(f"Train Precision:  {precision_train:.4f}")
print(f"Test Accuracy:    {accuracy_test:.4f}")
print(f"Test Precision:   {precision_test:.4f}")
print(f"Train Recall:     {recall_train:.4f}")
print(f"Test Recall:      {recall_test:.4f}")

print("=" * 60)
print("\n" + "=" * 80)
print("Random Forest Model Training Complete!")
print("=" * 80)
