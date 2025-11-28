
# ============================================================================
# STEP 12: TRAIN/TEST SPLIT
# ============================================================================
# Task 1: Restaurant Rating Prediction
# Split data into training set (80%) and testing set (20%)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================================
# ASSUMPTION: You already have X and y from STEP 11
# ============================================================================
# If not, run STEP 11 first to get:
# X = Feature matrix (7316 rows, 4 features)
# y = Target vector (7316 ratings)

# For reference, if you need to recreate them:
# X should have columns: ['Price range', 'Votes', 'Has_Table_Booking', 'Has_Online_Delivery']
# y should have: Aggregate ratings (1.8 - 4.9)

print("=" * 60)
print("STEP 12: TRAIN/TEST SPLIT")
print("=" * 60)
print()

# ============================================================================
# STEP 12.1: CHECK DATA BEFORE SPLIT
# ============================================================================

print("=" * 60)
print("DATA BEFORE SPLIT")
print("=" * 60)
print(f"Total samples: {len(X)}")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print()

print("Feature columns:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")
print()

print("Sample of features (X):")
print(X.head())
print()

print("Sample of target (y):")
print(y.head())
print()

# ============================================================================
# STEP 12.2: SPLIT DATA (80% TRAIN, 20% TEST)
# ============================================================================

print("=" * 60)
print("SPLITTING DATA: 80% TRAIN, 20% TEST")
print("=" * 60)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,                    # Features
    y,                    # Target
    test_size=0.2,        # 20% for testing
    random_state=42       # For reproducibility
)

print(f"\n✅ Split complete!")
print()

# ============================================================================
# STEP 12.3: CHECK SPLIT RESULTS
# ============================================================================

print("=" * 60)
print("SPLIT RESULTS")
print("=" * 60)

print("\nTRAINING SET:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Number of samples: {len(X_train)}")
print(f"  Percentage: {(len(X_train) / len(X)) * 100:.1f}%")

print("\nTESTING SET:")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")
print(f"  Number of samples: {len(X_test)}")
print(f"  Percentage: {(len(X_test) / len(X)) * 100:.1f}%")
print()

# ============================================================================
# STEP 12.4: VERIFY DATA DISTRIBUTION
# ============================================================================

print("=" * 60)
print("VERIFYING DATA DISTRIBUTION")
print("=" * 60)

print("\nTarget (Rating) Statistics:")
print("\nOriginal Data (y):")
print(f"  Mean: {y.mean():.2f}")
print(f"  Median: {y.median():.2f}")
print(f"  Std: {y.std():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

print("\nTraining Set (y_train):")
print(f"  Mean: {y_train.mean():.2f}")
print(f"  Median: {y_train.median():.2f}")
print(f"  Std: {y_train.std():.2f}")
print(f"  Min: {y_train.min():.2f}")
print(f"  Max: {y_train.max():.2f}")

print("\nTesting Set (y_test):")
print(f"  Mean: {y_test.mean():.2f}")
print(f"  Median: {y_test.median():.2f}")
print(f"  Std: {y_test.std():.2f}")
print(f"  Min: {y_test.min():.2f}")
print(f"  Max: {y_test.max():.2f}")
print()

# Check if distributions are similar
mean_diff = abs(y_train.mean() - y_test.mean())
if mean_diff < 0.1:
    print("✅ Train and Test distributions are similar! (Good split)")
else:
    print("⚠️ Train and Test distributions differ slightly")
print()

# ============================================================================
# STEP 12.5: VISUALIZE THE SPLIT
# ============================================================================

import matplotlib.pyplot as plt

print("=" * 60)
print("CREATING VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Training set distribution
axes[0].hist(y_train, bins=20, color='blue', edgecolor='black', alpha=0.7)
axes[0].axvline(y_train.mean(), color='red', linestyle='--', 
                label=f'Mean: {y_train.mean():.2f}')
axes[0].set_title('Training Set - Rating Distribution')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Testing set distribution
axes[1].hist(y_test, bins=20, color='green', edgecolor='black', alpha=0.7)
axes[1].axvline(y_test.mean(), color='red', linestyle='--', 
                label=f'Mean: {y_test.mean():.2f}')
axes[1].set_title('Testing Set - Rating Distribution')
axes[1].set_xlabel('Rating')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Visualization created!")
print()

# ============================================================================
# STEP 12.6: SUMMARY
# ============================================================================

print("=" * 60)
print("TRAIN/TEST SPLIT COMPLETE! ✅")
print("=" * 60)

print("\nWhat we have now:")
print(f"  📊 X_train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  📊 X_test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"  🎯 y_train: {len(y_train)} target values")
print(f"  🎯 y_test:  {len(y_test)} target values")

print("\nNext Steps:")
print("  1. Train Linear Regression model on X_train, y_train")
print("  2. Train Decision Tree model on X_train, y_train")
print("  3. Train Random Forest model on X_train, y_train")
print("  4. Test all models on X_test, y_test")
print("  5. Compare performance!")
print()

print("=" * 60)
print("Ready for STEP 13: MODEL TRAINING! 🚀")
print("=" * 60)

# ============================================================================
# REMEMBER FOR INTERVIEW:
# ============================================================================
"""
Q: "Why do we split data into train and test sets?"
A: "We split data to evaluate how well our model generalizes to unseen data.
   The training set (80%) is used to teach the model patterns, while the 
   testing set (20%) acts as 'new' data the model has never seen. This 
   prevents overfitting—where a model memorizes training data but fails on 
   new data. By testing on unseen data, we get an honest measure of the 
   model's real-world performance. I used an 80/20 split with random_state=42 
   for reproducibility."

Q: "What is random_state=42?"
A: "random_state is a seed for the random number generator. It ensures that 
   the split is reproducible—running the code multiple times will produce 
   the same train/test split. The value 42 is arbitrary (a reference to 
   'Hitchhiker's Guide to the Galaxy'), but any integer works. This is 
   important for debugging and comparing models fairly."
"""

