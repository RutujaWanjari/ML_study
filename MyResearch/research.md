### MICE (Multiple Imputation by Chained Equations)

1. This technique is used to find a relation between missing features and the in-hand features
2. This relationship can be linear or non-linear
3. Can be used where the features are important but missing, ex- surveys
4. To apply MICE, create 5 copies (say) of this simple data set and cycle multiple times through the steps below for each copy:
   1. **Step 1:** Replace (or impute) the missing values in each variable with temporary "place holder" values derived solely from the non-missing values available for that variable. For example, replace the missing age value with the mean age value observed in the data, replace the missing income values with the mean income value observed in the data, etc.
   2. **Step 2** Set back to missing the “place holder” imputations for the age variable only. This way, the current data copy contains missing values for age, but not for income and gender.
   3. **Step 3:** Regress age on income and gender via a linear regression model (though it is possible to also regress age on only one of these variables); to be able to fit the model to the current data copy, drop all the records where age is missing during the model fitting process. In this model, age is the dependent variable and income and gender are the independent variables.
   4. **Step 4** Use the fitted regression model in the previous step to predict the missing age values. (When age will be subsequently used as an independent variable in the regression models for other variables, both the observed values of age and these predicted values will be used.) The article doesn't make it clear that a random component should be added to these predictions.
   5. **Step 5:** Repeat Steps 2–4 separately for each variable that has missing data, namely income and gender.
5. Cycling through Steps 1 - 5 once for each of the variables age, income and gender constitutes one *cycle*. At the end of this cycle, all of the missing values in age, income an gender will have been replaced with predictions from regression models that reflect the relationships observed in the data between these variables.
6. "fancyimpute" is a library widely used for MICE, method used fancyimpute.MICE().complete(data matrix)
