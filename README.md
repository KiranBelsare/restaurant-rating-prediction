# Restaurant Rating Prediction

Machine Learning project predicting restaurant ratings using operational features.

## 🎯 Project Overview

Built and compared three ML models (Linear Regression, Decision Tree, Random Forest) to predict restaurant ratings. **Random Forest achieved R² = 0.44**, with Votes identified as the dominant predictor (86% feature importance).

## 📊 Dataset

- **Original Records:** 9,551 restaurants
- **After Cleaning:** 7,316 restaurants
- **Features:** 4 (Price range, Votes, Table Booking, Online Delivery)
- **Target:** Aggregate rating (1.8 - 4.9 scale)

## 🏆 Results

| Model | R² Score | RMSE | MAE | Performance |
|-------|----------|------|-----|-------------|
| **Random Forest** | **0.4408** | **0.4108** | **0.3006** | **Best** 🏆 |
| Decision Tree | 0.3955 | 0.4271 | 0.3092 | Good |
| Linear Regression | 0.3038 | 0.4584 | 0.3576 | Baseline |

## 🔍 Key Findings

- **Votes dominates predictions** (86% importance)
- Random Forest outperformed Linear Regression by **45%**
- Average prediction error: ±0.30 rating points

## 🛠️ Tech Stack

- **Python 3.x**
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn

## 📁 Project Structure
```
restaurant-rating-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── restaurant_data.csv
├── src/
│   ├── 01_data_exploration.py
│   ├── 02_data_cleaning.py
│   ├── 03_train_test_split.py
│   ├── 04_model_training.py
│   └── 05_feature_importance.py
└── docs/
    └── PROJECT_DOCUMENTATION.md
```

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/restaurant-rating-prediction.git](https://github.com/KiranBelsare/restaurant-rating-prediction.git)
cd restaurant-rating-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the pipeline:**
```bash
# Step 1: Data Exploration
python src/01_data_exploration.py

# Step 2: Data Cleaning
python src/02_data_cleaning.py

# Step 3: Train/Test Split
python src/03_train_test_split.py

# Step 4: Model Training
python src/04_model_training.py

# Step 5: Feature Importance
python src/05_feature_importance.py
```

## 📈 Model Performance

### Random Forest (Best Model)
- Explains 44% of rating variance
- Predicts within ±0.30 rating points on average
- 45% improvement over baseline

### Feature Importance
1. **Votes:** 85.72% (Dominant!)
2. **Price range:** 6.15%
3. **Online Delivery:** 5.18%
4. **Table Booking:** 2.95%

## 💡 Future Improvements

- Add cuisine type encoding
- Implement sentiment analysis on reviews
- Try XGBoost/LightGBM
- Add cross-validation

## 👨‍💻 Author

**Kiran Belsare**
- Email: kirabel.998@gmail.com
- LinkedIn: www.linkedin.com/in/kiran-belsare-a6b46128b
- GitHub: https://github.com/KiranBelsare

## 📄 License

MIT License

## 🙏 Acknowledgments

- Cognifyz Technologies - Internship opportunity
- Dataset source: Restaurant rating data
