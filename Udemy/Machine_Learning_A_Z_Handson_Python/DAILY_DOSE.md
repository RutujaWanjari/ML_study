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

### 17/05/2021

##### Natural Language Processing

1. Types of NLP
2. Bag of words model
3. NLP data cleaning pipeline
4. CountVectorizer
5. GuassianNaiveBayes classifier
6. Predicting new data after model is trained

### 22/03/2021

##### Classification

1. Linear algos - Logistic Regression, SVM
2. Non-linear algos - KNN, Kernel SVM, Random Forest

### 03/06/2021

##### Logistic Regresssion

### 14/06/2021

#### SVM

1. Some dataset are clearly separated from each other and we can create a straight LINE between them. These cases need linear algos like LR, LinearSVM(kernel='linear')
2. For non-linear datasets, we require non-linear algos like SVM(kernel='rbf'), K-nn, NN
3. Identify if data is linearly separable, depending on that choose algos (linear algos / non-linear algos)

### 26/06/2021

##### Kernel SVM

1. For kernel SVM, 1 technique for converting non-linear separable data to linear separable, add 1 more dimension to data
2. In single dimension, data is separated by dot, in 2D data is separated by line, in 3D data is separated by a plane.
3. The problem with this technique of adding 1 more dimension (mapping to a higher dimension) is that it is computationally very expensive because we need to create a mapping function/formula to convert n dimensional data to n+1 dimensional data.
4. we can create a decision boundary with very complex non linear data with guassion rbf

### 07/06/2021

##### Decision Tree

1. Decision Tree is a classification algorithm that was used in old days, but due to lots of better algos in new age it started fading out.
2. Then there were some updates made into it and Decision Tree algo can now again have proved useful with names like Random Forest, Isolation Forest, Gradient Boosting.

##### Random Forest

1. Ensemble Learning - Its a technique when more than 1 algos are commbined to get a higher performing algo
2. Ex of random forest usecase - [hand motion detection](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/BodyPartRecognition.pdf) and [video](https://www.microsoft.com/en-us/research/publication/real-time-human-pose-recognition-in-parts-from-a-single-depth-image/) of the same.

##### Classification Model Selection

##### Metrics

1. FP - We predicted it will occur, it actually did not
2. FN - We predicted it won't occur, it actually did
