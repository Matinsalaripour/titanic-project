# Titanic Survival Prediction - Project Report

## 📌 Project Overview
This project aims to predict the survival of passengers aboard the RMS Titanic using machine learning. The goal is to build a robust preprocessing pipeline and classification model that achieves high accuracy on the Kaggle test set.

## 🧹 Data Cleaning & Feature Engineering
- **Missing Values**: Imputed `Age` with median, `Embarked` with mode, and `Fare` with median (for test set).
- **Title Extraction**: Extracted titles (Mr, Mrs, Miss, Master, etc.) from the `Name` column to capture social status and age/gender nuances.
- **Family Features**: Created `FamilySize` (SibSp + Parch + 1) and `IsAlone` to capture group dynamics.
- **Age Binning**: Binned `Age` into 5 ordinal categories (Child, Teen, Adult, Middle_Aged, Elder) to capture non-linear U-shaped survival patterns.
- **Encoding**: Used One-Hot Encoding for `Title` and `Embarked`. Encoded `Sex` as binary (0=Female, 1=Male).
- **Dimensionality Reduction**: Dropped `Name`, `Ticket`, `Cabin`, `SibSp`, `Parch`, `AgeLog` (redundant), and `FamilySize` (weaker than `IsAlone`).

## 🛠️ Libraries Used
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn (preprocessing, model selection, metrics)
- **Boosting**: XGBoost

## 🤖 Models Evaluated
Three models were trained and validated using an 80/20 stratified split (with raw data split **before** feature engineering to prevent data leakage):

| Model | Validation Accuracy |
| :--- | :--- |
| Logistic Regression | `~0.78` |
| Random Forest | `~0.81` |
| **XGBoost (Tuned)** | **`~0.82`** |

## 📊 Key Insights (From Correlation Analysis)
1. **Sex is the strongest predictor**: Being female (0) dramatically increased survival chances.
2. **Class & Wealth**: 1st class passengers and those with higher fares had significantly higher survival rates.
3. **Title indicates survival**: `Mrs` (0.69) and `Miss` (~0.55) correlated highly with survival, while `Mr` correlated negatively.
4. **Family Structure**: Passengers traveling alone (`IsAlone` = 1) were less likely to survive.
5. **Age Effect**: The linear correlation with age was weak, but the binned age groups revealed that children (Master/Child) had higher survival rates.

## 📈 Final Results
- **Kaggle Public Leaderboard Score**: **0.77511** (or `77.5%` accuracy).
- **Best Model**: XGBoost with tuned hyperparameters (`max_depth=6`, `learning_rate=0.1`, `n_estimators=100`).

## 🚀 How to Reproduce
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the notebooks in order:
   - `01_EDA_and_Visualization.ipynb`
   - `02_Feature_Engineering.ipynb`
   - `03_Model_Selection.ipynb`
4. Or, run the Python scripts in `src/` directly to train and predict.

## 🔮 Future Improvements (Optional)
- **Ensemble Methods**: Combine XGBoost with Random Forest or a Neural Network for potentially higher accuracy.
- **Feature Engineering**: Extract the first letter of the `Cabin` column (e.g., 'A', 'B', 'C') to capture deck-level effects.
- **Hyperparameter Optimization**: Use Optuna or Bayesian Optimization for a more exhaustive search.
- **Imputation Strategy**: Test KNN Imputer for `Age` instead of median.

## 👤 Author
- **Matin Salari Pour**   
- **[Date: June 2026]**

---
*Project completed as part of a Data Science workflow practice.*