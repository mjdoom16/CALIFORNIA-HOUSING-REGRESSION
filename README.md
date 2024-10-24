# HousingRegressions

## Summary
After exploring the California Housing dataset provided by sklearn, key features were found before using a variety of supervised learning tasks to identify the best model. Then hyperparameter optimization to find the optimum hyperparameter value. The predicted vs actual (& residuals) targets were then plotted to see performance of the model.

## Exploratory Data Analysis
First, the dataset was evaluated to find information about features. All columns were floats, thus useful for a regression task. Histograms are a powerful tool to find distribution of data. AveX and Population showed long tails with almost all data located at the first few values. These features had extreme values as we evaluate the difference between the max and upper quartile values.

Interesting features were location coordinates, MedIncome and house age.

Using a scatter plot of the location long. & lat., we can visualise the bimodal distribution seen in the histogram before, indicating that a large proportion of homes are within these districts. Furthermore, the higher end of median house values were also located in these regions too.

Using a Pairplot, we can evaluate every possible xy combination of features, and we see no pattern on house value outside of median income, which implies that this is a relevant feature.

## Predictive Analysis
Dropping all irrelevant columns (keeping MedInc, Long., Lat.), we created a subset data for a regression task that would be able to predict median house values. 

Models used:
- Linear Regression
- Decision Tree
- Gradient Boosting
- Histogram Gradient Boosting
- Random Forest

Using R2 score and mean square error (MSE) as the scoring functions, each method was assessed using the same hyperparameters.

Hyperparameter:
- max_iter/n_estimators = 1000,
- max_depth = 3,
- min_samples_split = 3,
- learning_rate = 0.01,
- loss/criterion = squared_error
  
This showed Histogram Gradient Boosting (HGBT) as the best regression model, with a score of 0.809 and 0.255, respectively. HGBT being best suited for large datasets (n_sample >=10,000) due its algorithm having an operation time of O(n_feat * n), where n is the number of samples at a node, which is superior to regular GBT algorithm. Surprisingly, Random Forest (RF) performed the worst, with scores of 0.506 and 0.661, respectively

## Hyperparameter Optimization
After finding the model that performed the best, hyperparameters were tuned to tweak the model for optimum performance. The GridSearchCV was used, with parameter grid including:
- max_iter = [100,250,500,750,1000]
- max_depth = [1,3,5,7]
- learning_rate = [0.001, 0.01, 0.05]

using mse as the scoring function and a cross validation strategy of 5-fold. From this, we found the best score was (-)0.233, 
- max_iter = 750
- max_depth = 5
- learning_rate = 0.05

 which returned the same values as the inital model search, indicating hyperparameters could have been smaller for speed.

 ## Visualising predictions
 Using the PredictionErrorDisplay metric, two plots were created (True vs Pred & Residual vs Pred), with a 10-fold cross-validation on a subsample of 250 points.
  
 True vs Pred, showed a positive trend to the data, with points close to the best fit line, as evidenced by the low MSE score.
 Residuals vs Pred, shows how far predicted points stray, with most data found ± 0.5, implying the model fit well.

 ## Next Steps

 The next step would be building a neural network (NN) model for regression tasks. While linear models and decision trees (& ensemble methods) are good tools to use, we could also assess the performance of NNs.
