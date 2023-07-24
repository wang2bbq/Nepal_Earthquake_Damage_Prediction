# Modeling-Earthquake-Damage
Based on aspects of building location and construction, the goal is to predict the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal. There are 3 grades of the damage: 1 represents low damage, 2 represents a medium amount of damage, 3 represents almost complete destruction.

# Abstract
After setting up libraries and loading data, exploratory data analysis was conducted, which included checking for missing data, duplicated rows, outliers, and class imbalance. To address outliers, winsorization was performed. For feature engineering, categorical features were converted using one-hot encoding. In correlation analysis, two features with high correlation to others were dropped. An attempt was made to use PCA, but it was observed that each principal component had similar variance. Several models were experimented with, including Random Forest, XGBoost, LightGBM, Catboost, SVM, and Multinomial Logistic Regression. Based on the scores from stratified 5-fold cross-validation, GBDT models performed better. Hyperparameters for XGBoost, LightGBM, and Catboost were tuned using Optuna, and XGBoost outperformed LightGBM and Catboost on this dataset, achieving a micro-averaged F1 score of 0.7484. In order to further improve the performance of XGBoost, especially on the minority class, SMOTE was utilized for data pre-processing to address the class imbalance issue. While cross-validation results showed improvement, the performance on the test set did not improve significantly. To further enhance the model's performance, multiple ensemble methods were tried. Ultimately, a Soft Voting ensemble of 5 XGBoost models with different hyperparameters performed the best, achieving a micro F1 score of 0.7502.

# Exploratory Data Analysis
The dataset for this project consists of 260,601 examples in the training set, each with 38 features, including 8 categorical features. Additionally, there are 86,868 examples in the test set.
- missing values
- duplicated rows
- outliers
- class imbalance

# Feature Engineering
- one-hot encoding
- correlation analysis and feature selection
- standardization and PCA

# Model Selection
- Stratified k-fold cross-validation
  - random forest (0.6377)
  - XGBoost (0.7478)
  - LightGBM (0.7427)
  - CatBoost (0.7349)
  - SVM (0.5903)
  - multinomial logistic regression (0.5758)

# Parameter Tuning
- Optuna
  - XGBoost 1 (0.7484)
  - XGBoost 2 (0.7480)
  - XGBoost 3 (0.7479)
  - XGBoost 4 (0.7480)
  - XGBoost 5 (0.7481)
  - XGBoost + SMOTE 1 (0.7661)
  - XGBoost + SMOTE 2 (0.7665)
  - XGBoost + SMOTE 3 (0.7667)
  - LightGBM (0.7453)
  - CatBoost (0.7463)

# Classifier Comparation
- feature importance
- prediction comparation

# Ensemble Methods
- Hard Voting
- Soft Voting

# Future Work
More attention should be given to feature engineering, particularly for the three geographic region features: 'geo_level_1_id', 'geo_level_2_id', and 'geo_level_3_id'. An AutoEncoder can be employed to extract valuable information from these geographic ID features. The most specific location ID, 'geo_level_3_id', will be used as input, while the larger location IDs ['geo_level_1_id', 'geo_level_2_id'] will serve as output. The AutoEncoder will have just one hidden layer. Subsequently, these three features will be removed from the data, and the embedded features will be added to the training and test data.
