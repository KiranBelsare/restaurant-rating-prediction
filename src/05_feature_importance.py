# ============================================================================
# STEP 14: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
# Task 1: Restaurant Rating Prediction
# Discover which features matter MOST for predictions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# ASSUMPTION: You have trained models from STEP 13
# ============================================================================
# You should have: lr_model, dt_model, rf_model

print("=" * 60)
print("STEP 14: FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)
print()

print("Analyzing which features matter most for predictions...")
print()

# ============================================================================
# STEP 14.1: LINEAR REGRESSION COEFFICIENTS
# ============================================================================

print("=" * 60)
print("LINEAR REGRESSION: FEATURE COEFFICIENTS")
print("=" * 60)
print()

# Get coefficients
lr_coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_,
    'Abs_Coefficient': np.abs(lr_model.coef_)
})

# Sort by absolute value (importance)
lr_coefficients = lr_coefficients.sort_values('Abs_Coefficient', ascending=False)

print("Feature Coefficients (sorted by importance):")
print(lr_coefficients.to_string(index=False))
print()

print("Interpretation:")
print("  - Positive coefficient → Higher feature value = Higher rating")
print("  - Negative coefficient → Higher feature value = Lower rating")
print("  - Larger absolute value = More important feature")
print()

# ============================================================================
# STEP 14.2: DECISION TREE FEATURE IMPORTANCE
# ============================================================================

print("=" * 60)
print("DECISION TREE: FEATURE IMPORTANCE")
print("=" * 60)
print()

# Get feature importance
dt_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_model.feature_importances_
})

# Sort by importance
dt_importance = dt_importance.sort_values('Importance', ascending=False)

print("Feature Importance (sorted by importance):")
print(dt_importance.to_string(index=False))
print()

print("Interpretation:")
print("  - Importance ranges from 0 to 1")
print("  - Higher value = More important for splitting decisions")
print("  - Sum of all importances = 1.0")
print()

# ============================================================================
# STEP 14.3: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================

print("=" * 60)
print("RANDOM FOREST: FEATURE IMPORTANCE")
print("=" * 60)
print()

# Get feature importance
rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})

# Sort by importance
rf_importance = rf_importance.sort_values('Importance', ascending=False)

print("Feature Importance (sorted by importance):")
print(rf_importance.to_string(index=False))
print()

print("Interpretation:")
print("  - Importance ranges from 0 to 1")
print("  - Average importance across 100 trees")
print("  - More stable than single Decision Tree")
print()

# ============================================================================
# STEP 14.4: COMPARE FEATURE IMPORTANCE ACROSS ALL MODELS
# ============================================================================

print("=" * 60)
print("FEATURE IMPORTANCE COMPARISON")
print("=" * 60)
print()

# Combine all importances
comparison_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Linear_Regression': np.abs(lr_model.coef_),
    'Decision_Tree': dt_model.feature_importances_,
    'Random_Forest': rf_model.feature_importances_
})

# Normalize Linear Regression coefficients to 0-1 scale for comparison
lr_normalized = comparison_df['Linear_Regression'] / comparison_df['Linear_Regression'].sum()
comparison_df['Linear_Regression_Normalized'] = lr_normalized

print("Feature Importance Across All Models:")
print(comparison_df[['Feature', 'Linear_Regression_Normalized', 'Decision_Tree', 'Random_Forest']].to_string(index=False))
print()

# Find most important feature for each model
print("Most Important Feature per Model:")
for col in ['Linear_Regression_Normalized', 'Decision_Tree', 'Random_Forest']:
    most_important = comparison_df.loc[comparison_df[col].idxmax(), 'Feature']
    importance_value = comparison_df[col].max()
    model_name = col.replace('_Normalized', '').replace('_', ' ')
    print(f"  {model_name}: {most_important} ({importance_value:.4f})")
print()

# ============================================================================
# STEP 14.5: VISUALIZE FEATURE IMPORTANCE
# ============================================================================

print("=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Linear Regression Coefficients
colors_lr = ['green' if x > 0 else 'red' for x in lr_coefficients['Coefficient']]
axes[0, 0].barh(lr_coefficients['Feature'], lr_coefficients['Coefficient'], 
                color=colors_lr, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=0.8)
axes[0, 0].set_xlabel('Coefficient Value')
axes[0, 0].set_title('Linear Regression: Feature Coefficients\n(Green=Positive, Red=Negative)')
axes[0, 0].grid(True, axis='x', alpha=0.3)

# Add value labels
for i, (feature, coef) in enumerate(zip(lr_coefficients['Feature'], lr_coefficients['Coefficient'])):
    axes[0, 0].text(coef, i, f' {coef:.4f}', va='center', fontsize=9)

# Plot 2: Decision Tree Importance
axes[0, 1].barh(dt_importance['Feature'], dt_importance['Importance'], 
                color='blue', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Decision Tree: Feature Importance')
axes[0, 1].grid(True, axis='x', alpha=0.3)

# Add value labels
for i, (feature, imp) in enumerate(zip(dt_importance['Feature'], dt_importance['Importance'])):
    axes[0, 1].text(imp, i, f' {imp:.4f}', va='center', fontsize=9)

# Plot 3: Random Forest Importance
axes[1, 0].barh(rf_importance['Feature'], rf_importance['Importance'], 
                color='orange', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Random Forest: Feature Importance')
axes[1, 0].grid(True, axis='x', alpha=0.3)

# Add value labels
for i, (feature, imp) in enumerate(zip(rf_importance['Feature'], rf_importance['Importance'])):
    axes[1, 0].text(imp, i, f' {imp:.4f}', va='center', fontsize=9)

# Plot 4: Comparison of All Models
x = np.arange(len(X_train.columns))
width = 0.25

axes[1, 1].bar(x - width, comparison_df['Linear_Regression_Normalized'], 
               width, label='Linear Regression', color='green', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x, comparison_df['Decision_Tree'], 
               width, label='Decision Tree', color='blue', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x + width, comparison_df['Random_Forest'], 
               width, label='Random Forest', color='orange', alpha=0.7, edgecolor='black')

axes[1, 1].set_xlabel('Features')
axes[1, 1].set_ylabel('Importance (Normalized)')
axes[1, 1].set_title('Feature Importance: All Models Comparison')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(X_train.columns, rotation=45, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Visualizations created!")
print()

# ============================================================================
# STEP 14.6: KEY INSIGHTS
# ============================================================================

print("=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print()

# Most important feature overall (from Random Forest)
most_important_feature = rf_importance.iloc[0]['Feature']
most_important_value = rf_importance.iloc[0]['Importance']

print(f"🏆 MOST IMPORTANT FEATURE: {most_important_feature}")
print(f"   Importance: {most_important_value:.4f} ({most_important_value*100:.2f}%)")
print()

# Least important feature
least_important_feature = rf_importance.iloc[-1]['Feature']
least_important_value = rf_importance.iloc[-1]['Importance']

print(f"📉 LEAST IMPORTANT FEATURE: {least_important_feature}")
print(f"   Importance: {least_important_value:.4f} ({least_important_value*100:.2f}%)")
print()

# Feature ranking consensus
print("Feature Ranking (from Random Forest):")
for i, row in rf_importance.iterrows():
    feature = row['Feature']
    importance = row['Importance']
    percentage = importance * 100
    print(f"  {i+1}. {feature}: {importance:.4f} ({percentage:.2f}%)")
print()

# ============================================================================
# STEP 14.7: RECOMMENDATIONS
# ============================================================================

print("=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print()

print("Based on feature importance analysis:")
print()

# Top features
top_features = rf_importance.head(2)['Feature'].tolist()
print(f"✅ KEEP THESE FEATURES (Most important):")
for feature in top_features:
    imp = rf_importance[rf_importance['Feature'] == feature]['Importance'].values[0]
    print(f"   - {feature} (Importance: {imp:.4f})")
print()

# Low importance features
low_features = rf_importance.tail(2)['Feature'].tolist()
print(f"⚠️ CONSIDER REMOVING (Least important):")
for feature in low_features:
    imp = rf_importance[rf_importance['Feature'] == feature]['Importance'].values[0]
    print(f"   - {feature} (Importance: {imp:.4f})")
    print(f"     → Contributes only {imp*100:.2f}% to predictions")
print()

print("💡 SUGGESTIONS FOR IMPROVEMENT:")
print("   1. Focus data collection on top features")
print("   2. Consider engineering new features related to top performers")
print("   3. Could simplify model by removing low-importance features")
print("   4. Investigate why certain features are more predictive")
print()

# ============================================================================
# STEP 14.8: SUMMARY
# ============================================================================

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE! ✅")
print("=" * 60)
print()

print("Summary:")
print(f"  ✅ Analyzed feature importance across 3 models")
print(f"  ✅ Most important: {most_important_feature}")
print(f"  ✅ Least important: {least_important_feature}")
print(f"  ✅ Visualizations created")
print()

print("Next Steps:")
print("  📝 STEP 15: Final documentation and summary")
print("  🎓 STEP 16: Prepare for interviews")
print()

# ============================================================================
# REMEMBER FOR INTERVIEW:
# ============================================================================
"""
Q: "Which features were most important for your model?"
A: "Using Random Forest's feature importance, I found that [Votes] 
   was the most significant predictor with an importance of [0.8572], contributing 
   85% to the model's decisions. This makes intuitive sense because [explain why]. 
   [Has table booking] had minimal impact at only [2.95] importance, 
   suggesting it could potentially be removed without significant performance loss. 
   Interestingly, all three models agreed that [top feature] was crucial, which 
   validates its importance across different algorithmic approaches."

Q: "How did you measure feature importance?"
A: "I used three approaches: For Linear Regression, I examined coefficient 
   magnitudes, where larger absolute values indicate stronger influence. For 
   Decision Tree and Random Forest, I used Gini importance, which measures how 
   much each feature reduces impurity when making splits. Random Forest averages 
   this across 100 trees, providing more stable importance scores. I visualized 
   all three to ensure consistency and found strong agreement on the top features."
"""