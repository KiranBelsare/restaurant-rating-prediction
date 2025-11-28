# ============================================================================
# STEP 11: DATA CLEANING & FEATURE ENGINEERING
# ============================================================================
# Task 1: Restaurant Rating Prediction
# This step prepares data for machine learning modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (if not already loaded)
df = pd.read_csv('restaurant_data.csv')

print("=" * 60)
print("ORIGINAL DATA")
print("=" * 60)
print(f"Total Restaurants: {len(df)}")
print(f"Total Features: {df.shape[1]}")
print()

# ============================================================================
# STEP 11.1: REMOVE RESTAURANTS WITH RATING = 0
# ============================================================================
# Why? Ratings of 0 are usually fake/closed restaurants or data errors

print("=" * 60)
print("STEP 11.1: REMOVING RATING = 0")
print("=" * 60)

# Count how many have rating 0
zero_ratings = (df['Aggregate rating'] == 0).sum()
print(f"Restaurants with rating 0: {zero_ratings}")
print(f"Percentage: {(zero_ratings / len(df)) * 100:.2f}%")
print()

# Remove them
df_clean = df[df['Aggregate rating'] > 0].copy()

print(f"After removal:")
print(f"Total Restaurants: {len(df_clean)}")
print(f"Removed: {len(df) - len(df_clean)}")
print()

# ============================================================================
# STEP 11.2: REMOVE EXTREME OUTLIERS IN COST
# ============================================================================
# Why? Some costs are 0 (impossible) or 800,000 (extreme outliers)

print("=" * 60)
print("STEP 11.2: REMOVING COST OUTLIERS")
print("=" * 60)

print(f"Before removal:")
print(f"Min Cost: {df_clean['Average Cost for two'].min()}")
print(f"Max Cost: {df_clean['Average Cost for two'].max()}")
print(f"Mean Cost: {df_clean['Average Cost for two'].mean():.2f}")
print()

# Remove restaurants with cost = 0 (impossible)
df_clean = df_clean[df_clean['Average Cost for two'] > 0].copy()

# Remove extreme outliers (cost > 99th percentile)
# 99th percentile means: keep only bottom 99% of costs
percentile_99 = df_clean['Average Cost for two'].quantile(0.99)
print(f"99th percentile cost: {percentile_99:.2f}")

df_clean = df_clean[df_clean['Average Cost for two'] <= percentile_99].copy()

print(f"\nAfter removal:")
print(f"Total Restaurants: {len(df_clean)}")
print(f"Min Cost: {df_clean['Average Cost for two'].min()}")
print(f"Max Cost: {df_clean['Average Cost for two'].max()}")
print(f"Mean Cost: {df_clean['Average Cost for two'].mean():.2f}")
print()

# ============================================================================
# STEP 11.3: HANDLE MISSING VALUES IN CUISINES
# ============================================================================
# Why? We found 0.09% missing cuisines earlier

print("=" * 60)
print("STEP 11.3: HANDLING MISSING CUISINES")
print("=" * 60)

missing_cuisines = df_clean['Cuisines'].isnull().sum()
print(f"Missing cuisines: {missing_cuisines}")

if missing_cuisines > 0:
    # Option 1: Remove rows with missing cuisines (simplest)
    df_clean = df_clean[df_clean['Cuisines'].notna()].copy()
    print(f"Removed {missing_cuisines} rows with missing cuisines")
    print(f"Total Restaurants now: {len(df_clean)}")
else:
    print("No missing cuisines to handle!")
print()

# ============================================================================
# STEP 11.4: ENCODE BINARY FEATURES (Yes/No → 1/0)
# ============================================================================
# Why? ML models need numbers, not text

print("=" * 60)
print("STEP 11.4: ENCODING BINARY FEATURES")
print("=" * 60)

# Has Table booking: Yes → 1, No → 0
df_clean['Has_Table_Booking'] = (df_clean['Has Table booking'] == 'Yes').astype(int)

# Has Online delivery: Yes → 1, No → 0
df_clean['Has_Online_Delivery'] = (df_clean['Has Online delivery'] == 'Yes').astype(int)

print("Encoded features:")
print(f"Has_Table_Booking: {df_clean['Has_Table_Booking'].value_counts().to_dict()}")
print(f"Has_Online_Delivery: {df_clean['Has_Online_Delivery'].value_counts().to_dict()}")
print()

# ============================================================================
# STEP 11.5: SELECT FINAL FEATURES FOR MODELING
# ============================================================================
# Based on correlation analysis, we'll use:
# - Price range (0.44 correlation) ✅
# - Votes (0.31 correlation) ✅
# - Has_Table_Booking (new binary feature) ✅
# - Has_Online_Delivery (new binary feature) ✅

print("=" * 60)
print("STEP 11.5: SELECTING FEATURES")
print("=" * 60)

# Define feature columns
feature_columns = [
    'Price range',
    'Votes',
    'Has_Table_Booking',
    'Has_Online_Delivery'
]

# Target column
target_column = 'Aggregate rating'

# Create feature matrix (X) and target vector (y)
X = df_clean[feature_columns].copy()
y = df_clean[target_column].copy()

print("Selected Features:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")
print()
print(f"Target Variable: {target_column}")
print()

print(f"Feature Matrix (X) shape: {X.shape}")
print(f"Target Vector (y) shape: {y.shape}")
print()

# ============================================================================
# STEP 11.6: CHECK FINAL CLEANED DATA
# ============================================================================

print("=" * 60)
print("FINAL CLEANED DATA SUMMARY")
print("=" * 60)

print(f"Total Restaurants: {len(X)}")
print(f"Total Features: {len(feature_columns)}")
print()

print("Feature Statistics:")
print(X.describe())
print()

print("Target Statistics:")
print(y.describe())
print()

# ============================================================================
# STEP 11.7: VISUALIZE CLEANED DATA
# ============================================================================

print("=" * 60)
print("CREATING VISUALIZATIONS OF CLEANED DATA")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Distribution of ratings (after cleaning)
axes[0, 0].hist(y, bins=20, color='green', edgecolor='black')
axes[0, 0].set_title('Distribution of Ratings (After Cleaning)')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.2f}')
axes[0, 0].legend()

# Plot 2: Price Range vs Rating
X_with_target = X.copy()
X_with_target['Rating'] = y
X_with_target.boxplot(column='Rating', by='Price range', ax=axes[0, 1])
axes[0, 1].set_title('Rating by Price Range (Cleaned)')
axes[0, 1].set_xlabel('Price Range')
axes[0, 1].set_ylabel('Rating')

# Plot 3: Votes vs Rating
axes[1, 0].scatter(X['Votes'], y, alpha=0.5, color='blue')
axes[1, 0].set_title('Votes vs Rating (Cleaned)')
axes[1, 0].set_xlabel('Number of Votes')
axes[1, 0].set_ylabel('Rating')

# Plot 4: Feature Correlation Heatmap
X_with_target = X.copy()
X_with_target['Aggregate rating'] = y
correlation_matrix = X_with_target.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 11.8: SAVE COMPARISON STATISTICS
# ============================================================================

print("=" * 60)
print("BEFORE vs AFTER CLEANING COMPARISON")
print("=" * 60)

print(f"\nOriginal Data:")
print(f"  Restaurants: {len(df)}")
print(f"  Rating range: {df['Aggregate rating'].min():.1f} - {df['Aggregate rating'].max():.1f}")
print(f"  Average rating: {df['Aggregate rating'].mean():.2f}")
print(f"  Median rating: {df['Aggregate rating'].median():.2f}")

print(f"\nCleaned Data:")
print(f"  Restaurants: {len(y)}")
print(f"  Rating range: {y.min():.1f} - {y.max():.1f}")
print(f"  Average rating: {y.mean():.2f}")
print(f"  Median rating: {y.median():.2f}")

print(f"\nData Removed:")
print(f"  Total removed: {len(df) - len(y)}")
print(f"  Percentage removed: {((len(df) - len(y)) / len(df)) * 100:.2f}%")
print()

print("=" * 60)
print("DATA CLEANING COMPLETE! ✅")
print("=" * 60)
print(f"\nReady for modeling with:")
print(f"  - {len(X)} restaurants")
print(f"  - {len(feature_columns)} features")
print(f"  - Clean data (no 0 ratings, no extreme outliers)")
print()

# ============================================================================
# REMEMBER FOR INTERVIEW:
# ============================================================================
"""
Q: "How did you clean the data?"
A: "I performed several cleaning steps:
   1. Removed restaurants with rating 0 (fake/closed entries)
   2. Removed cost outliers (0 cost and values above 99th percentile)
   3. Removed rows with missing cuisines (only 9 rows)
   4. Encoded binary features (Yes/No to 1/0)
   5. Selected 4 key features based on correlation analysis
   This reduced the dataset from 9,551 to ~7,000 restaurants with clean, 
   model-ready data."
"""

