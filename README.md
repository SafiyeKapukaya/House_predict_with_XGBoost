This is a house price predict model.I deleted the columns with more than half missing values ​​and filled in the missing values. 
Then I combined the two datasets. 
Encode operations have been completed.
Finally, it was divided into training and test data and log transformation was applied to the independent variable.
The model was trained with the XGBoost algorithm on the train set. 
rmse : 0.1322
