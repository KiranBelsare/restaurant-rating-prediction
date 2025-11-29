# LESSON 1: LOADING AND EXPLORING DATA
# Task 1: Restaurant Rating Prediction

# ============================================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================================
# Think of libraries as toolkits. We're importing tools we'll need.

import pandas as pd           # For data manipulation (reading CSV, cleaning, etc.)
import numpy as np            # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations (graphs, charts)
import seaborn as sns         # For prettier visualizations

# ============================================================================
# STEP 2: LOAD THE DATA
# ============================================================================
# This reads your CSV file into a DataFrame (like Excel table in Python)

# OPTION A: If you have the file locally (save it in same folder as this code)
df = pd.read_csv('data/restaurant_data.csv')

# OPTION B: If file is too big or you want to load sample
# df = pd.read_csv('data/restaurant_data.csv', nrows=100)  # Load only first 100 rows

# ============================================================================
# STEP 3: FIRST LOOK AT DATA
# ============================================================================

# How many rows and columns?
print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(f"Total Rows (Restaurants): {df.shape[0]}")
print(f"Total Columns (Features): {df.shape[1]}")
print()

# See first 5 rows
print("=" * 60)
print("FIRST 5 ROWS OF DATA")
print("=" * 60)
print(df.head())
print()

# See all column names
print("=" * 60)
print("ALL COLUMNS")
print("=" * 60)
print(df.columns.tolist())
print()

# ============================================================================
# STEP 4: DATA TYPES
# ============================================================================
# Understanding what type of data each column has

print("=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)
print()

# ============================================================================
# STEP 5: STATISTICAL SUMMARY
# ============================================================================
# Get mean, median, min, max for numerical columns

print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())
print()

# ============================================================================
# STEP 6: MISSING VALUES
# ============================================================================
# Check if any data is missing (NaN = Not a Number)

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing_count = df.isnull().sum()
print(missing_count[missing_count > 0])  # Only show columns with missing values
print()

# Percentage of missing values
print("=" * 60)
print("PERCENTAGE OF MISSING VALUES")
print("=" * 60)
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0])
print()

# ============================================================================
# STEP 7: FOCUS ON TARGET VARIABLE (Aggregate Rating)
# ============================================================================
# This is what we're trying to predict!

print("=" * 60)
print("TARGET VARIABLE: AGGREGATE RATING")
print("=" * 60)
print(f"Minimum Rating: {df['Aggregate rating'].min()}")
print(f"Maximum Rating: {df['Aggregate rating'].max()}")
print(f"Average Rating: {df['Aggregate rating'].mean():.2f}")
print(f"Median Rating: {df['Aggregate rating'].median():.2f}")
print()

# How many restaurants have each rating?
print("=" * 60)
print("RATING DISTRIBUTION")
print("=" * 60)
print(df['Aggregate rating'].value_counts().sort_index())
print()

# ============================================================================
# STEP 8: EXPLORE KEY FEATURES
# ============================================================================

print("=" * 60)
print("KEY FEATURES ANALYSIS")
print("=" * 60)

# Average Cost for Two
print("\n1. AVERAGE COST FOR TWO")
print(f"   Min: {df['Average Cost for two'].min()}")
print(f"   Max: {df['Average Cost for two'].max()}")
print(f"   Mean: {df['Average Cost for two'].mean():.2f}")
print(f"   Missing: {df['Average Cost for two'].isnull().sum()}")

# Price Range
print("\n2. PRICE RANGE")
print(f"   Unique values: {df['Price range'].unique()}")
print(f"   Distribution:\n{df['Price range'].value_counts().sort_index()}")

# Has Table Booking
print("\n3. HAS TABLE BOOKING")
print(f"   Yes: {(df['Has Table booking'] == 'Yes').sum()}")
print(f"   No: {(df['Has Table booking'] == 'No').sum()}")

# Has Online Delivery
print("\n4. HAS ONLINE DELIVERY")
print(f"   Yes: {(df['Has Online delivery'] == 'Yes').sum()}")
print(f"   No: {(df['Has Online delivery'] == 'No').sum()}")

# Votes
print("\n5. VOTES")
print(f"   Min: {df['Votes'].min()}")
print(f"   Max: {df['Votes'].max()}")
print(f"   Mean: {df['Votes'].mean():.2f}")
print(f"   Missing: {df['Votes'].isnull().sum()}")

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Distribution of Ratings
axes[0, 0].hist(df['Aggregate rating'], bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Restaurant Ratings')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Price Range vs Rating (Box Plot)
df.boxplot(column='Aggregate rating', by='Price range', ax=axes[0, 1])
axes[0, 1].set_title('Rating by Price Range')
axes[0, 1].set_xlabel('Price Range')
axes[0, 1].set_ylabel('Rating')

# Plot 3: Votes vs Rating (Scatter Plot)
axes[1, 0].scatter(df['Votes'], df['Aggregate rating'], alpha=0.5)
axes[1, 0].set_title('Votes vs Rating')
axes[1, 0].set_xlabel('Number of Votes')
axes[1, 0].set_ylabel('Rating')

# Plot 4: Average Cost vs Rating (Scatter Plot)
axes[1, 1].scatter(df['Average Cost for two'], df['Aggregate rating'], alpha=0.5, color='orange')
axes[1, 1].set_title('Average Cost vs Rating')
axes[1, 1].set_xlabel('Average Cost for Two')
axes[1, 1].set_ylabel('Rating')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 10: CORRELATION ANALYSIS
# ============================================================================
# Which numerical features correlate with rating?

print("=" * 60)
print("CORRELATION WITH RATING")
print("=" * 60)

# Select only numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numerical_cols].corr()['Aggregate rating'].sort_values(ascending=False)
print(correlation)
print()

# Visualize correlation
plt.figure(figsize=(10, 6))
correlation.plot(kind='barh')
plt.title('Feature Correlation with Aggregate Rating')
plt.xlabel('Correlation Coefficient')
plt.show()

# ============================================================================
# KEY INSIGHTS (What We Learned)
# ============================================================================

print("=" * 60)
print("KEY INSIGHTS FROM EXPLORATION")
print("=" * 60)
print("""
1. Target Variable: Ratings range from X to Y (fill in from data)
2. Most Important Features:
   - Feature A has correlation 0.XX (fill from correlation analysis)
   - Feature B has correlation 0.XX
3. Missing Data: X% of data is missing (fill from analysis)
4. Next Step: Clean data and prepare for modeling
""")

# ============================================================================
# REMEMBER FOR INTERVIEW:
# ============================================================================
"""
Q: "What did you learn from exploratory data analysis?"
A: "I explored the dataset and found:
   - X restaurants with Y features
   - Ratings range from 0 to 5
   - Most restaurants are priced at range 3
   - Votes and Price Range correlate with ratings
   - Z% missing data in [column name]
   - This helped me understand which features to use for modeling"
"""
