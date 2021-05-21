### 26/04/2021

##### Setup

1. Git clone
2. Conda create py39env
3. dev branch

##### Udemy ML course intro

1. Facebook ML apps
2. Google collab
3. Why ML?
4. ML vs AI vs DL
5. Regression - Simple, Multiple, Polynomial

### 28/04/2021

##### Data preprocessing

1. Tools
2. DataPreprocessing numpy, pandas, matplotlib.plt, scikitlearn pip install
3. pandas fixed width file import like csv or json import
4. Data exploration before handling missing data to get which values are missing
5. sklearn.impute for handling missing values - [https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute](scikitlearn.impute)
6. imputer.fit() + imputer.transform() == imputer.fit_transform()
7. sklearn.compose estimator for categorical encoding independent variables - [https://scikit-learn.org/stable/modules/classes.html?highlight=sklearn%20compose#module-sklearn.compose](transformers)
8. **Question** - Feature scaling is applied before splitting dfata or after?
   1. AFTER
   2. FEATURE SCALING IS ALWAYS APPLIED **AFTER** SPLITTING TRAIN AND TEST DATASET
9. Data needs to be splitted for training and testing, so that trained model can be used to test whole new unseen data.
10. sklearn.model_selection for train_test_split() - [https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection](https://)
11. Feature scaling is required so that some features do not dominate other features. Not all ML models require feature scaling.
12. Standardization vs Normalization - SN can be used all the time, NN can be used when all the features are normally distributed.
13. **Question** - Do we apply feature scaling on dummy features (features that we encoded) ?
    1. NO
    2. The reason being - purpose of feature_scaling is to normalize all the features, which means feature value = +/-3 (in standardization) and 0/1 (in normalization)
    3. As the dummy feature values are already 0 or 1, we do not perfrom feature scaling on them

### 17/03/2021

##### Natural Language Processing

1. Types of NLP
2. Bag of words model
3. NLP data cleaning pipeline
4. CountVectorizer
5. GuassianNaiveBayes classifier
6. Predicting new data after model is trained
