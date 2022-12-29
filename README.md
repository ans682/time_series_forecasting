# timeseries_forecasting
This is a package for time series forecasting.

It contains six base models for univariate and multivariate time series forecasting. The package also contains a Random Forest Regressor ensemble method which aggregates predictions from the selected base models to output a new time series prediction.

## Overview of the files
Below is the overview of the `src` folder. It contains the following folders and files:
- `base_models` folder contains base forecasting models.
  - `ar.py` contains AutoRegression (AR) model. An AR model is a type of time series forecasting model that uses previous values of a given variable to predict its future values. The model assumes that the future value of a variable is directly related to its past values, and that the relationship between the variable and its past values can be described by a linear equation. This equation is then used to make predictions about the future values of the variable. To build an AutoRegression model, we start by selecting the number of previous values of the variable that we want to use in our model (this is called the "lag" of the model, and we call it `num_features` in our package). We then create a linear regression model using these past values as predictors and the variable itself as the target variable. The coefficients of the linear regression model are used to make predictions about the future values of the variable. The `ar.py` file contains two functions: `ar_bike_predict()` and `ar_predict()`, one for predicting number of bikes demanded and the other for predicting stock returns respectively.
    - `ar_bike_predict()` has the following arguments: `train` dataset, `test` dataset, `output_path` where the predictions and their plot are saved, `timeseries_name` which is used to name the plot, `sgdr_parameters` which contains AR's parameters defined in the YAML file (See description of YAML file at the end of the README). The `ar_bike_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data. 
      2. Create MinMaxScaler with feature_range=(0,1). We use scaler in order to reduce the range of dataset's values and improve forecasting accuracy of the model. Fit the MinMaxScaler to the `total_data` dataset. Transform `total_data` and convert it to Python list.
      3. Create lists to store total_x_features and total_y_features. For instance, let `num_features` = 25. Let `total_data` dataset have 1000 entries. Slice first 25 entries from `total_data` and append them to the `total_x_features`. In other words, add `total_data[0:num_features]` to the `total_x_features` list. Then slide a window by one entry forward and add it to the list, i.e. add `total_data[1:num_features + 1]` to the `total_x_features` list. Fill out the `total_x_features` list until you reach the end of the dataset. `total_y_features` list contains values which are located right after our sliding window in the `total_data`. For example, the 0th value in the `total_y_features` list equals `total_data[num_features]`, the 1st value equals `total_data[num_features + 1]`, etc. 
      4. Extract training data from the `total_x_features` and `total_y_features` lists.
      5. Create an instance of SGDRegressor. We use scikit-learn Stochastic Gradient Descent Regressor because it fulfills the job of AutoRegression model in terms of fitting a linear model. The linear model is fitted by minimizing loss with the Stochastic Gradient Descent. `shuffle` parameter is set to be False because we don't want to shuffle our time series data. Sequence of data does matter in our case. `learning_rate` parameters is taken from the YAML file.
      6. Fit SGDRegressor to the training data.
      7. Loop through test data and make prediction at every iteration. Append prediction to the 'predictions' list. Then fit the SGDRegressor to a dataset with the current test data entry.
      8. Inverse transform the predictions by using `inverse_transform()`.
      9. Calculate r2-score by using the produced predictions and test ground truth values.
      10. Plot graph with the predictions and test ground truth values. Save the plot in the output_path directory.
      11. Save predictions into a CSV file. Save test ground truth values into a CSV file as well.
      12. Return predictions, test values and r2-score.
    - `ar_predict()` has the same arguments as `ar_bike_predict()`. This function is used for making predictions for financial time series such as stock data. The `ar_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data.
      2. Difference `total_data`, i.e. calculate stock returns for every day. 
      3. Repeat steps from 3-7 and from 9-12 from `ar_bike_predict()`.


  - `arima.py` contains AutoRegressive Integrated Moving Average (ARIMA) model. The ARIMA model is specified by three parameters: ’p’, ’d’, and ’q’. The ’p’ parameter is the order of the autoregressive part of the model, which specifies the number of previous values of the time series that are used to predict its future values. The ’d’ parameter is the degree of differencing, which specifies the number of times the time series has been differenced in order to make it stationary. The ’q’ parameter is the order of the moving average part of the model, which specifies the number of past forecast errors that are used to predict the future values of the time series. 
  To build an ARIMA model, we start by selecting the ap- propriate values for the ’p’, ’d’, and ’q’ parameters. We then estimate the coefficients of the model using the selected val- ues of the parameters, and use the estimated coefficients to make predictions about the future values of the time series. 
  The `arima.py` file contains two functions: `arima_bike_predict()` and `arima_predict()`, one for predicting number of bikes demanded and the other for predicting stock returns respectively.
    - `arima_bike_predict()` has the following arguments: `train` dataset, `test` dataset, `output_path` where the predictions and their plot are saved, `timeseries_name` which is used to name the plot, `arima_parameters` which contains ARIMA parameters defined in the YAML file (See description of YAML file at the end of the README). The `arima_bike_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data. 
      2. Create MinMaxScaler with feature_range=(0,1). We use scaler in order to reduce the range of dataset's values and improve forecasting accuracy of the model. Fit the MinMaxScaler to the `total_data` dataset. Transform `total_data` and convert it to Python list.
      3. Loop through range of test data. Fit ARIMA model with data up to current test data entry (exclusively). Use `p`,`d`,`q` parameters from the YAML file. Make a forecast and append it to the predictions list.
      4. Inverse transform the predictions by using `inverse_transform()`.
      5. Calculate r2-score by using the produced predictions and test ground truth values.
      6. Plot graph with the predictions and test ground truth values. Save the plot in the output_path directory.
      7. Save predictions into a CSV file. Save test ground truth values into a CSV file as well.
      8. Return predictions, test values and r2-score.
    - `arima_predict()` has the same arguments as `arima_bike_predict()`. This function is used for making predictions for financial time series such as stock data. The `arima_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data.
      2. Difference `total_data`, i.e. calculate stock returns for every day. 
      3. Repeat step 3 and steps from 5-8 from `arima_bike_predict()`.

  - `theta.py` contains Theta model. The Theta model is a type of time series forecasting model that is based on the idea of exponential smoothing. It is a simple and intuitive method that can be used to forecast the future values of a time series, and it is particularly useful for data with a trend or seasonal component.
  The file contains two functions: `theta_bike_predict()` and `theta_predict()`, one for predicting number of bikes demanded and the other for predicting stock returns respectively.
    - `theta_bike_predict()` has the following arguments: `train` dataset, `test` dataset, `output_path` where the predictions and their plot are saved, `timeseries_name` which is used to name the plot, `theta_parameters` which contains Theta parameters defined in the YAML file (See description of YAML file at the end of the README). The `theta_bike_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data. 
      2. Create MinMaxScaler with feature_range=(0,1). We use scaler in order to reduce the range of dataset's values and improve forecasting accuracy of the model. Fit the MinMaxScaler to the `total_data` dataset. Transform `total_data` and convert it to Python list.
      3. Loop through range of test data. Fit Theta model with data up to current test data entry (exclusively). Set `deseasonalize` argument to the parameter specified in the YAML file. Make a forecast and append it to the predictions list.
      4. Inverse transform the predictions by using `inverse_transform()`.
      5. Calculate r2-score by using the produced predictions and test ground truth values.
      6. Plot graph with the predictions and test ground truth values. Save the plot in the output_path directory.
      7. Save predictions into a CSV file. Save test ground truth values into a CSV file as well.
      8. Return predictions, test values and r2-score.
    - `theta_predict()` has the same arguments as `theta_bike_predict()`. This function is used for making predictions for financial time series such as stock data. The `theta_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data.
      2. Difference `total_data`, i.e. calculate stock returns for every day. 
      3. Repeat step 3 and steps from 5-8 from `theta_bike_predict()`.

  - `var.py` contains Vector AutoRegression (VAR) model. Vector autoregression (VAR) is a type of time series forecast- ing model that is used to model the relationship between multiple time series. It is a generalization of an Autoregres- sion model, which is used to model the relationship between a single time series. 
  In a VAR model, each time series is modeled as a linear function of its own past values, as well as the past values of the other time series in the model. This allows us to capture the interdependencies between the time series, and to make forecasts that take into account the effects of one time series on the others.
  To build a VAR model, we start by selecting the number of previous values of each time series that we want to use in the model (this is called the "lag" of the model). We then create a linear regression model using these past values as predictors, and the current values of the time series as the target variables. The coefficients of the linear regression model are used to make predictions about the future values of the time series.
  The file contains two functions: `var_bike_predict()` and `var_predict()`, one for predicting number of bikes demanded and the other for predicting stock returns respectively.
    - `var_bike_predict()` has the following arguments: `train` dataset, `test` dataset, `output_path` where the predictions and their plot are saved, `timeseries_name` which is used to name the plot, `var_parameters` which contains Theta parameters defined in the YAML file (See description of YAML file at the end of the README). The `var_bike_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data. 
      2. Create two MinMaxScalers with feature_range=(0,1) for two different time series. Fit the MinMaxScaler to two time series' datasets. Transform two scaled time series datasets and convert them to Python lists. Join the lists into a Python dictionary. Convert the dictionary into `total_data` DataFrame.
      3. Create VAR model. Fit the model to training data with the `p` parameter from the YAML file.
      4. Loop through range of test data. Fit VAR model with data up to current test data entry (exclusively). Set `p` argument to the parameter specified in the YAML file. Make a forecast and append it to the predictions list.
      5. Inverse transform the predictions by using `inverse_transform()`.
      6. Calculate r2-score by using the produced predictions and test ground truth values.
      7. Plot graph with the predictions and test ground truth values. Save the plot in the output_path directory.
      8. Save predictions into a CSV file. Save test ground truth values into a CSV file as well.
      9. Return predictions, test values and r2-score.
    - `var_predict()` has the same arguments as `var_bike_predict()`. This function is used for making predictions for financial time series such as stock data. The `var_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data.
      2. Difference `total_data`, i.e. calculate stock returns for every day for two time series.
      3. Repeat steps 3-4 and steps 6-9 from `var_bike_predict()`.

  - `rfr.py` contains Random Forest Regressor (RFR) model. Random Forest Regressor is a type of machine learning al- gorithm that can be used for time series forecasting. It is a type of ensemble model, which means that it combines the predictions of multiple individual models in order to make more accurate forecasts.
  In a Random Forest Regressor, the individual models are decision trees. A decision tree is a type of model that uses a tree-like structure to make predictions. Each node in the tree represents a decision about the value of a predictor variable, and each branch represents the possible outcomes of that decision. The leaves of the tree represent the final predictions made by the model.
  To build a Random Forest Regressor model for time series forecasting, we start by dividing the time series into a number of overlapping time windows. For each time window, we use the values in the time window as predictors, and the next value in the time series as the target variable. We then train a decision tree model on each time window, using the values in the time window as predictors and the next value in the time series as the target. 
  Once all of the decision tree models have been trained, we combine their predictions to create the final forecast. This is typically done by taking the average of the predictions made by the individual decision tree models.
  The file contains two functions: `rfr_bike_predict()` and `rfr_predict()`, one for predicting number of bikes demanded and the other for predicting stock returns respectively.
    - `rfr_bike_predict()` has the following arguments: `train` dataset, `test` dataset, `output_path` where the predictions and their plot are saved, `timeseries_name` which is used to name the plot, `rfr_parameters` which contains RFR's parameters defined in the YAML file (See description of YAML file at the end of the README). The `rfr_bike_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data. 
      2. Create MinMaxScaler with feature_range=(0,1). We use scaler in order to reduce the range of dataset's values and improve forecasting accuracy of the model. Fit the MinMaxScaler to the `total_data` dataset. Transform `total_data` and convert it to Python list.
      3. Create lists to store total_x_features and total_y_features. For instance, let `num_features` = 25. Let `total_data` dataset have 1000 entries. Slice first 25 entries from `total_data` and append them to the `total_x_features`. In other words, add `total_data[0:num_features]` to the `total_x_features` list. Then slide a window by one entry forward and add it to the list, i.e. add `total_data[1:num_features + 1]` to the `total_x_features` list. Fill out the `total_x_features` list until you reach the end of the dataset. `total_y_features` list contains values which are located right after our sliding window in the `total_data`. For example, the 0th value in the `total_y_features` list equals `total_data[num_features]`, the 1st value equals `total_data[num_features + 1]`, etc. 
      4. Extract training data from the `total_x_features` and `total_y_features` lists.
      5. Create an instance of RandomForestRegressor (RFR). Set `bootstrap` parameter to the value specified in the YAML file. 
      6. Fit RFR model to the training data.
      7. Loop through test data. At every iteration, fit the RFR model to a dataset up to the current test data entry. Then make a prediction and append it to the 'predictions' list. 
      8. Inverse transform the predictions by using `inverse_transform()`.
      9. Calculate r2-score by using the produced predictions and test ground truth values.
      10. Plot graph with the predictions and test ground truth values. Save the plot in the output_path directory.
      11. Save predictions into a CSV file. Save test ground truth values into a CSV file as well.
      12. Return predictions, test values and r2-score.
    - `rfr_predict()` has the same arguments as `rfr_bike_predict()`. This function is used for making predictions for financial time series such as stock data. The `rfr_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data.
      2. Difference `total_data`, i.e. calculate stock returns for every day. 
      3. Repeat steps from 3-7 and from 9-12 from `rfr_bike_predict()`.

  - `lstm.py` contains Long Short-Term Memory (LSTM) model.
  This model is a type of recurrent neural network that is well-suited for modeling time series data. Unlike traditional feedforward neural networks, LSTM networks can retain previous information in their internal state, allowing them to effectively incorporate long-term dependencies in the data. The LSTM model works by dividing the input sequence into individual time steps, each of which is passed through a layer of LSTM cells. Each LSTM cell takes in a time step and an optional input from the previous cell, and outputs a new representation of the time step that includes information from the past. This output is then passed on to the next LSTM cell in the sequence. At each time step, the LSTM cell updates its internal state by forgetting some information, adding new information, and outputting a new representation of the input. This allows the LSTM network to effectively retain long-term dependen- cies in the data and make predictions based on previous events in the time series. To make predictions with an LSTM model, the model is provided with a sequence of historical data and asked to predict the next value in the sequence. The model uses its internal state to incorporate information from the past and make a prediction for the future.
  The `lstm.py` file contains two functions: `lstm_bike_predict()` and `lstm_predict()`, one for predicting number of bikes demanded and the other for predicting stock returns respectively.
    - `lstm_bike_predict()` has the following arguments: `train` dataset, `test` dataset, `output_path` where the predictions and their plot are saved, `timeseries_name` which is used to name the plot, `lstm_parameters` which contains LSTM parameters defined in the YAML file (See description of YAML file at the end of the README). The `lstm_bike_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data. 
      2. Create MinMaxScaler. We use scaler in order to reduce the range of dataset's values and improve forecasting accuracy of the model. Fit the MinMaxScaler to the full dataset. Transform scaled dataset.
      3. Iterate through the values to create combinations of X samples. 
      4. Reshape the input as a 3D (number of samples, time steps, number of features). Do not reshape `y` as a 3D data as it is supposed to be a single column only.
      5. Split the data into train and test.
      6. Initialize the RNN by creating an instance of Sequential().
      7. Create the first (input) hidden layer with the parameters from the YAML file.
      8. Create the second hidden layer with the parameters from the YAML file.
      9. Create the third hidden layer with the parameters from the YAML file.
      10. Create the output layer with the parameters from the YAML file.
      11. Compile the RNN.
      12. Loop through test data. At every 20 iterations, fit the LSTM model to a dataset up to the current test data entry. Then make a prediction. Transform it by using `inverse_transform()` and append it to the 'predictions' list. 
      13. Calculate r2-score by using the produced predictions and test ground truth values.
      14. Plot graph with the predictions and test ground truth values. Save the plot in the output_path directory.
      15. Save predictions into a CSV file. Save test ground truth values into a CSV file as well.
      16. Return predictions, test values and r2-score.
    - `lstm_predict()` has the same arguments as `lstm_bike_predict()`. This function is used for making predictions for financial time series such as stock data. The `lstm_predict()` function does the following:
      1. Concatenate `train` and `test` datasets into a single DataFrame object called `total_data`, which will be later used to scale current data.
      2. Difference `total_data`, i.e. calculate stock returns for every day. 
      3. Repeat steps from 3-16 from `lstm_bike_predict()`.
  
- `data_processing`
  - `load_bike.py` file contains `create_bike_df()` function which is used to load bike dataset. If user set is_bike_ts_valid=True, i.e. their time series has valid format, then the time series is simply loaded from the CSV by using pandas built-in read_csv() method. Otherwise, bike dataset preprocessing workflow is executed. To see the valid format of the time series, check one of the sections below.
  
  - `load_stock.py` file contains `create_stock_df()` function which is used to load stock dataset.

- `datasets` folder contains sample CSV files with bike and stock time series. Feel free to use these datasets to see how the package works.

- `ensemble_methods`
  - `rf_ensemble.py` file contains a Random Forest Regressor ensemble method. The file contains two functions: `rfr_ensemble_bike_predict()` and `rfr_ensemble_predict()`, one for predicting number of bikes demanded and the other for predicting stock returns respectively.
    - `rfr_ensemble_bike_predict()` has the following arguments: `train` dataset, `test` dataset, `output_path` where the predictions and their plot are saved, `timeseries_name` which is used to name the plot, `base_model_predictions` which is a dictionary with the predictions from the selected base models. The `rfr_ensemble_bike_predict()` function does the following:
      1. Convert `base_model_predictions` into `base_models_df` DataFrame.
      2. Create MinMaxScaler with feature_range=(0,1). We use scaler in order to reduce the range of dataset's values and improve forecasting accuracy of the model. Fit the MinMaxScaler to the `base_models_df` dataset. Transform `base_models_df`.
      3. Create `meta_X` list and fill it out with the predictions from the `base_models_df` up to the end of train dataset's size. We do this step in order to convert predictions into a vector which could be input into the Random Forest Regressor. 
      4. Create Random Forest Regressor model and fit it into the meta_X vector and ground truth values
      5. Loop through test data. At every iteration, fit the RFR model to a dataset up to the current test data entry. Then make a prediction and append it to the 'predictions' list. 
      6. Inverse transform the predictions by using `inverse_transform()`.
      7. Calculate r2-score by using the produced predictions and test ground truth values.
      8. Plot graph with the predictions and test ground truth values. Save the plot in the output_path directory.
      9. Save predictions into a CSV file. Save test ground truth values into a CSV file as well.
      10. Return predictions, test values and r2-score.
    - `rfr_ensemble_predict()` has the same arguments as `rfr_ensemble_bike_predict()`. This function is used for making predictions for financial time series such as stock data. The `rfr_ensemble_predict()` function does the following:
      1. Difference `total_data`, i.e. calculate stock returns for every day. 
      2. Repeat steps from 2-10 from `rfr_ensemble_bike_predict()`.

- `helpers`
  - `helpers.py` file contains functions which are used by the data_processing methods.

## YAML file
You can change the default parameters of the base forecasting models inside the `data.yaml` file. Specify the input_path to your time series and output_path in which predictions and their plots will be saved. Indicate the input_file_name which is used to name the plots. If you're working with raw CitiBike datasets (the ones which were downloaded directly from the CitiBike website), then specify the names of the bike stations which you want to use for forecasting. The package will make predictions for the first bike station that you indicate in the YAML file. In addition, specify the exact names of the raw CSV files, if you want to input several months of bicycle data.
<b>IMPORTANT:</b> Make sure that you specify the correct input_file_name and input_path to your time series file in the `data.yaml` file.

## Format of input time series
The valid format for time series is as follows:
Date,Value1,Value2\
timestamp1,10,1.0\
timestamp2,20,2.0\
timestamp3,30,3.0

The first column called 'Date' contains timestamps (year, month, day, hour, or minute). 'Date' values are of string data type.
The second column contains numeric data (int or float) of first time series.
The third column contains numeric data (int or float) of second time series.

## How to run package
1. Download package.
2. Edit YAML file. Change parameters of base models if necessary. Specify input and output paths, name of the CSV file with your time series.
3. Open terminal window. 
4. Change directory to this package.
5. Run main.py
6. You will see a welcome message with the list of base models of this package.
7. Answer questions by typing either 'yes' or 'no' in the command line.
8. You can find predictions in your output_path.




