
# ============================================================================
# STEP 13: TRAIN 3 MACHINE LEARNING MODELS
# ============================================================================
# Task 1: Restaurant Rating Prediction
# We'll train: Linear Regression, Decision Tree, Random Forest

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time
import pickle                      # ⬅️ built-in, no install needed
from pathlib import Path           # ⬅️ to make sure models/ folder exists

# ============================================================================
# LOAD TRAIN/TEST SPLIT FROM PREVIOUS STEP
# ============================================================================

X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print("=" * 60)
print("STEP 13: TRAINING MACHINE LEARNING MODELS")
print("=" * 60)
print()

print(f"Training data: {len(X_train)} samples")
print(f"Testing data: {len(X_test)} samples")
print(f"Features: {list(X_train.columns)}")
print()


# ============================================================================
# STEP 13.1: TRAIN MODEL 1 - LINEAR REGRESSION
# ============================================================================

print("=" * 60)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 60)
print()

print("Training Linear Regression...")
start_time = time.time()

# Create the model
lr_model = LinearRegression()

# Train the model (fit = learn from data)
lr_model.fit(X_train, y_train)

# Calculate training time
train_time_lr = time.time() - start_time

print(f"✅ Training complete! Time: {train_time_lr:.3f} seconds")
print()

# Make predictions
print("Making predictions on test set...")
y_pred_lr = lr_model.predict(X_test)

print(f"✅ Predictions complete!")
print()

# Show the learned formula
print("Learned Formula:")
print(f"Rating = {lr_model.intercept_:.4f}", end="")
for feature, coef in zip(X_train.columns, lr_model.coef_):
    sign = "+" if coef >= 0 else "-"
    print(f" {sign} {abs(coef):.4f}×{feature}", end="")
print()
print()

# Calculate metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print("Performance Metrics:")
print(f"  MSE (Mean Squared Error):  {mse_lr:.4f}")
print(f"  RMSE (Root MSE):           {rmse_lr:.4f}")
print(f"  R² Score:                  {r2_lr:.4f}")
print(f"  MAE (Mean Absolute Error): {mae_lr:.4f}")
print()

# ============================================================================
# STEP 13.2: TRAIN MODEL 2 - DECISION TREE
# ============================================================================

print("=" * 60)
print("MODEL 2: DECISION TREE REGRESSOR")
print("=" * 60)
print()

print("Training Decision Tree...")
start_time = time.time()

# Create the model
dt_model = DecisionTreeRegressor(
    random_state=42,      # For reproducibility
    max_depth=10,         # Limit tree depth (prevents overfitting)
    min_samples_split=20  # Minimum samples to split a node
)

# Train the model
dt_model.fit(X_train, y_train)

# Calculate training time
train_time_dt = time.time() - start_time

print(f"✅ Training complete! Time: {train_time_dt:.3f} seconds")
print()

# Make predictions
print("Making predictions on test set...")
y_pred_dt = dt_model.predict(X_test)

print(f"✅ Predictions complete!")
print()

# Tree information
print("Tree Information:")
print(f"  Tree depth: {dt_model.get_depth()}")
print(f"  Number of leaves: {dt_model.get_n_leaves()}")
print()

# Calculate metrics
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)

print("Performance Metrics:")
print(f"  MSE (Mean Squared Error):  {mse_dt:.4f}")
print(f"  RMSE (Root MSE):           {rmse_dt:.4f}")
print(f"  R² Score:                  {r2_dt:.4f}")
print(f"  MAE (Mean Absolute Error): {mae_dt:.4f}")
print()

# ============================================================================
# STEP 13.3: TRAIN MODEL 3 - RANDOM FOREST
# ============================================================================

print("=" * 60)
print("MODEL 3: RANDOM FOREST REGRESSOR")
print("=" * 60)
print()

print("Training Random Forest (this may take a minute)...")
start_time = time.time()

# Create the model
rf_model = RandomForestRegressor(
    n_estimators=100,     # Number of trees
    random_state=42,      # For reproducibility
    max_depth=15,         # Limit tree depth
    min_samples_split=20, # Minimum samples to split
    n_jobs=-1             # Use all CPU cores (faster!)
)

# Train the model
rf_model.fit(X_train, y_train)

# Calculate training time
train_time_rf = time.time() - start_time

print(f"✅ Training complete! Time: {train_time_rf:.3f} seconds")
print()

# Make predictions
print("Making predictions on test set...")
y_pred_rf = rf_model.predict(X_test)

print(f"✅ Predictions complete!")
print()

# Forest information
print("Forest Information:")
print(f"  Number of trees: {rf_model.n_estimators}")
print(f"  Features used: {rf_model.n_features_in_}")
print()

# Calculate metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("Performance Metrics:")
print(f"  MSE (Mean Squared Error):  {mse_rf:.4f}")
print(f"  RMSE (Root MSE):           {rmse_rf:.4f}")
print(f"  R² Score:                  {r2_rf:.4f}")
print(f"  MAE (Mean Absolute Error): {mae_rf:.4f}")
print()

# ============================================================================
# STEP 13.4: COMPARE ALL 3 MODELS
# ============================================================================

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print()

# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'R² Score': [r2_lr, r2_dt, r2_rf],
    'RMSE': [rmse_lr, rmse_dt, rmse_rf],
    'MAE': [mae_lr, mae_dt, mae_rf],
    'Training Time (s)': [train_time_lr, train_time_dt, train_time_rf]
})

# Sort by R² score (higher is better)
comparison = comparison.sort_values('R² Score', ascending=False)

print(comparison.to_string(index=False))
print()

# Find best model
best_model_name = comparison.iloc[0]['Model']
best_r2 = comparison.iloc[0]['R² Score']

print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")
print()

# ============================================================================
# STEP 13.5: VISUALIZE MODEL PREDICTIONS
# ============================================================================

print("=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Actual vs Predicted - Linear Regression
axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.5, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Rating')
axes[0, 0].set_ylabel('Predicted Rating')
axes[0, 0].set_title(f'Linear Regression\nR² = {r2_lr:.4f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted - Decision Tree
axes[0, 1].scatter(y_test, y_pred_dt, alpha=0.5, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Rating')
axes[0, 1].set_ylabel('Predicted Rating')
axes[0, 1].set_title(f'Decision Tree\nR² = {r2_dt:.4f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Actual vs Predicted - Random Forest
axes[1, 0].scatter(y_test, y_pred_rf, alpha=0.5, color='orange')
axes[1, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual Rating')
axes[1, 0].set_ylabel('Predicted Rating')
axes[1, 0].set_title(f'Random Forest\nR² = {r2_rf:.4f}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Model Comparison Bar Chart
models = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest']
r2_scores = [r2_lr, r2_dt, r2_rf]
colors = ['blue', 'green', 'orange']

bars = axes[1, 1].bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('R² Score')
axes[1, 1].set_title('Model Performance Comparison')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Visualizations created!")
print()

# ============================================================================
# STEP 13.6: SAVE SAMPLE PREDICTIONS
# ============================================================================

print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)
print()

# Create comparison dataframe with first 10 predictions
sample_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'LR_Pred': y_pred_lr[:10],
    'DT_Pred': y_pred_dt[:10],
    'RF_Pred': y_pred_rf[:10]
})

sample_df['LR_Error'] = abs(sample_df['Actual'] - sample_df['LR_Pred'])
sample_df['DT_Error'] = abs(sample_df['Actual'] - sample_df['DT_Pred'])
sample_df['RF_Error'] = abs(sample_df['Actual'] - sample_df['RF_Pred'])

print("First 10 Predictions:")
print(sample_df.to_string(index=False))
print()

# ============================================================================
# STEP 13.7: SUMMARY
# ============================================================================

print("=" * 60)
print("TRAINING COMPLETE! ✅")
print("=" * 60)
print()

print("Summary:")
print(f"  ✅ Trained 3 models successfully")
print(f"  ✅ Best model: {best_model_name} (R² = {best_r2:.4f})")
print(f"  ✅ All predictions generated")
print(f"  ✅ Visualizations created")
print()

print("Next Steps:")
print("  📊 STEP 14: Detailed model evaluation")
print("  🔍 STEP 15: Feature importance analysis")
print("  📝 STEP 16: Final documentation")
print()

# ============================================================================
# STEP 13.8: SAVE TRAINED MODELS FOR LATER (FEATURE IMPORTANCE, DEPLOYMENT, ETC.)
# ============================================================================

print("=" * 60)
print("SAVING TRAINED MODELS")
print("=" * 60)

# Ensure the models directory exists
Path("models").mkdir(exist_ok=True)

# Save each model as a .pkl file
with open("models/lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("models/dt_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("Saved models to:")
print("  models/lr_model.pkl")
print("  models/dt_model.pkl")
print("  models/rf_model.pkl")
print()


# ============================================================================
# REMEMBER FOR INTERVIEW:
# ============================================================================
"""
Q: "Walk me through your model training process."
A: "I trained three regression models—Linear Regression, Decision Tree, and 
   Random Forest—on 5,852 training samples. Linear Regression served as a 
   baseline and finished in under 1 second. Decision Tree added complexity 
   to capture non-linear patterns. Random Forest, using 100 trees, provided 
   ensemble learning benefits. I used random_state=42 for reproducibility 
   and set max_depth and min_samples_split parameters to prevent overfitting. 
   After training, I evaluated all three on 1,464 unseen test samples using 
   R², RMSE, and MAE. Random Forest achieved the best R² of [0.4408], 
   outperforming Linear Regression by [Y%]."
"""
