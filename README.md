# Modeling-Earthquake-Damage
Based on aspects of building location and construction, the goal is to predict the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal.

# Abstract
I first conducted exploratory data analysis, which included checking for missing data, duplicated rows, outliers, and class imbalance. I performed winsorization to deal with outliers. For feature engineering, I converted categorical features using one-hot encoding. In correlation analysis, I dropped two features with high correlation to others. I also tried PCA but observed that each principal component had similar variance. I experimented with several models, including random forest, XGBoost, SVM, and so on. I focused on XGBoost and performed parameter tuning to optimize its performance. I also used SMOTE to improve the performance on the minority class. Finally, I achieved a micro averaged F1 score of 0.7502.

# Exploratory data analysis
The dataset for this project consists of 260,601 examples in the training set, each with 38 features, including 8 categorical features. Additionally, there are 86,868 examples in the test set.

- missing values
- duplicated rows
- outliers
- class imbalance

# Feature engineering
- one-hot encoding
- correlation analysis and feature selection
- standardization and PCA

# Model selection
- random forest (0.6377)
- XGBoost (0.7478)
- LightGBM (0.7427)
- CatBoost (0.7349)
- SVM (0.5903)
- multinomial logistic regression (0.5758)

# Parameter tuning
- - Optuna
- XGBoost 1 (0.7484)
- XGBoost 2 (0.7480)
- XGBoost + SMOTE 1 (0.7479)
- XGBoost + SMOTE 2 (0.7480)
- XGBoost + SMOTE 3 (0.7481)
- LightGBM (0.7453)
- CatBoost (0.7463)

# Classifier Comparation
- feature importance
- prediction comparation

# Ensemble methods
- Hard Voting
- Soft Voting

# Submission
After training XGBoost with the best parameters on the entire training set, I obtained a micro-averaged F1 score of 75.02, which is slightly lower than the 0.7518 achieved using the validation set after splitting the training set. However, upon closer inspection of the test set, I discovered that there were some previously overlooked details, such as 266 values of 'geo_level_3_id' that were not present in the training set. One solution to this problem is to treat these values equally as an "Unknown" category.

Additionally, I attempted to pre-process the training set using SMOTE, but this actually resulted in a lower score of 0.7492. This may be due to the fact that SMOTE adds noise to the data, which can have a negative impact on model performance.

Based on the analysis of feature importance, I have found that different models tend to focus on different parts of the feature space. Therefore, I believe that combining these models could lead to better performance. Moving forward, I plan to explore the use of more models and consider stacking or ensemble methods, such as mixture of experts, in order to further improve the results.

# Code
To run the code, a GPU is required since I am using the 'gpu_hist' tree method in XGBClassifier. I completed this project using Google Colab, so it is important to pay attention to the file path when loading data.

