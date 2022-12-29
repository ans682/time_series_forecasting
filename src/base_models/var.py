from statsmodels.tsa.api import VAR
import pandas as pd
from ..helpers.helpers import difference
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
##################################################################################################################################

##################################################################################################################################
def var_bike_predict(train, test, output_plots_path, timeseries_name, p):
    train_raw = train
    test_raw = test

    train_size = train.shape[0]
    test_size = test.shape[0]

    # Join train and test
    ### total_data = train.tolist() + test.tolist() # This is the merged list now. IF UNIVARIATE TS
    total_data = pd.concat([train, test], axis=0)
    total_data = total_data[['Value1', 'Value2']]

    ######### SECTION FOR RESCALING THE TOTAL DATASET ###########
    total_data_size = total_data.shape[0]
    print('Size of total data: ', len(total_data))
    print('Type of total data: ', type(total_data))

    # Get 'Value1' column's values
    total_data_val1 = total_data['Value1'].values.reshape(total_data_size, 1)
    total_data_val2 = total_data['Value2'].values.reshape(total_data_size, 1)
    # total_data = pd.concat([total_data_val1, total_data_val2], axis=1)

    # Create scaler
    scaler_val1 = MinMaxScaler(feature_range=(0, 1))
    scaler_val2 = MinMaxScaler(feature_range=(0, 1))

    # Fit data into scaler
    scaler_val1 = scaler_val1.fit(total_data_val1)
    scaler_val2 = scaler_val2.fit(total_data_val2)

    print('Scaler 1, Min: %f, Max: %f' % (scaler_val1.data_min_, scaler_val1.data_max_))
    print('Scaler 2, Min: %f, Max: %f' % (scaler_val2.data_min_, scaler_val2.data_max_))
    # Transform the dataset 
    total_data_val1 = scaler_val1.transform(total_data_val1).reshape(-1).tolist()
    total_data_val2 = scaler_val1.transform(total_data_val2).reshape(-1).tolist()

    # Create a df from two lists
    intermediate_dictionary = {'Value1':total_data_val1, 'Value2':total_data_val2}

    # Convert dictionary to Pandas dataframe
    total_data = pd.DataFrame(intermediate_dictionary)
    # total_data = pd.concat([total_data_val1, total_data_val2], axis=1)
    print('Total shape: ', total_data.shape)
    ##############################################################

    # total_data['Value2'] = total_data['Value2'].astype(int)


    # Difference time series, i.e. use returns
    ### total_data = difference(total_data) # IF UNIVARIATE TS
    ### total_size = len(total_data) # IF UNIVARIATE TS
    ### print('Size of differenced total data: ', len(total_data)) # IF UNIVARIATE TS

    # Save original data in order to be able to convert differenced values back to the original values.
    # test_start_idx = total_data.shape[0] - test_size
    # original_val1 = total_data['Value1'][test_start_idx]
    original_val1 = test_raw['Value1'].iloc[0]
    
    print('Orig test data: ', original_val1)

    ### Difference total_data
    # total_data = total_data.diff().dropna()
    # differenced_val1 = total_data['Value1'].iloc[1:]

    # print('------')
    # print(differenced_val1)
    # print('------')
    print('Orig val: ',original_val1)
    # print('Shape of differenced data: ',differenced_val1.shape)
    

    delta_train_df = total_data[:-test_size]
    delta_test_df = total_data[-test_size:]
    differenced_val1_column = delta_test_df['Value1']
    print('Shape of differenced data: ',differenced_val1_column.shape)
    print("Shape of delta Test: ", delta_test_df.shape)
    print("Shape of delta Train: ", delta_train_df.shape)
    print('Type of delta: ', type(delta_train_df))
    print('Head of delta: ', delta_train_df.head())
    print('name of columns: ', list(delta_train_df.columns))
    print('Data types of cols in delta df: ', delta_train_df.dtypes)

    predictions = []
    num_obs = 1 # Number of observations
    
    # Capture start time
    start_time = time.time()

    model = VAR(delta_train_df) 
    model_fitted = model.fit(p)
    lag_order = model_fitted.k_ar


    for i in range(0, test_size, num_obs):
        sliding_train_df = pd.concat([delta_train_df.iloc[i:], delta_test_df.iloc[:i]], axis=0)
        assert sliding_train_df.shape == delta_train_df.shape

        model = VAR(sliding_train_df)
        model_fitted = model.fit(p)

        forecast_input = sliding_train_df.values[-lag_order:]
        # Make a forecast
        forecast_output = model_fitted.forecast(y=forecast_input, steps=num_obs)
        print('Forecast: ', forecast_output)
        print('Type of forecast: ', type(forecast_output))
        # Append forecast_output to forecast_list
        predictions.append(forecast_output.tolist())
    
    # Capture end time
    end_time = time.time()
    total_time = end_time - start_time
    print("Time executed: ", total_time)
    # print("Size of predictions: ", len(predictions))
    print(len(predictions))
    value1_predictions = []
    for pair in predictions:
        value1_predictions.append(pair[0][0])
    print(len(value1_predictions))

    predictions = value1_predictions
    # test_diff = delta_test_df['Value1'].tolist()
    # print('Test diff shape: ', len(test_diff))

    test_diff = test_raw['Value1'].tolist()
############################################################

    # De-difference predictions
    # predictions = np.r_[original_val1, predictions].cumsum().astype(int)
    # predictions = predictions.tolist()

    # Convert predictions from list into numpy array. Then reshape into 2d array.
    predictions_size = len(predictions)
    predictions = np.array(predictions).reshape(predictions_size, 1)

    # Inverse transform predictions
    predictions = scaler_val1.inverse_transform(predictions)

    # Convert predictions to list
    predictions = predictions.reshape(-1).tolist()

    
############################################################


    # Calculate the R2 score using the predictions and the true values
    score = r2_score(test_diff, predictions)

    # Plot the predictions and the true values
    plt.figure(figsize=(20,10),dpi=300)
    plt.plot(test_diff, label="true")
    plt.plot(predictions, label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Number of bikes demanded', fontsize=16)
    title_name = 'Predicting VAR model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/VAR-' + timeseries_name + '.png')
    # plt.show()

    # Save predictions into CSV
    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/VAR-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/VAR-true_values.csv', index=False)

    return predictions, test_diff, score

##################################################################################################################################

##################################################################################################################################
def var_predict(train, test, output_plots_path, timeseries_name, p):
    train_raw = train
    test_raw = test

    train_size = train.shape[0]
    test_size = test.shape[0]

    # Join train and test
    ### total_data = train.tolist() + test.tolist() # This is the merged list now. IF UNIVARIATE TS
    total_data = pd.concat([train, test], axis=0)
    total_data = total_data[['Value1', 'Value2']]
    total_data['Value2'] = total_data['Value2'].astype(int)

    # Difference time series, i.e. use returns
    ### total_data = difference(total_data) # IF UNIVARIATE TS
    ### total_size = len(total_data) # IF UNIVARIATE TS
    ### print('Size of differenced total data: ', len(total_data)) # IF UNIVARIATE TS

    total_data = total_data.diff().dropna()
    delta_train_df = total_data[:-test_size]
    delta_test_df = total_data[-test_size:]
    print("Shape of delta Test: ", delta_test_df.shape)
    print("Shape of delta Train: ", delta_train_df.shape)
    print('Type of delta: ', type(delta_train_df))
    print('Head of delta: ', delta_train_df.head())
    print('name of columns: ', list(delta_train_df.columns))
    print('Data types of cols in delta df: ', delta_train_df.dtypes)

    predictions = []
    num_obs = 1 # Number of observations
    
    # Capture start time
    start_time = time.time()

    model = VAR(delta_train_df) 
    model_fitted = model.fit(p)
    lag_order = model_fitted.k_ar


    for i in range(0, test_size, num_obs):
        sliding_train_df = pd.concat([delta_train_df.iloc[i:], delta_test_df.iloc[:i]], axis=0)
        assert sliding_train_df.shape == delta_train_df.shape

        model = VAR(sliding_train_df)
        model_fitted = model.fit(p)

        forecast_input = sliding_train_df.values[-lag_order:]
        # Make a forecast
        forecast_output = model_fitted.forecast(y=forecast_input, steps=num_obs)
        print('Forecast: ', forecast_output)
        print('Type of forecast: ', type(forecast_output))
        # Append forecast_output to forecast_list
        predictions.append(forecast_output.tolist())
    
    # Capture end time
    end_time = time.time()
    total_time = end_time - start_time
    print("Time executed: ", total_time)
    # print("Size of predictions: ", len(predictions))
    print(len(predictions))
    value1_predictions = []
    for pair in predictions:
        value1_predictions.append(pair[0][0])
    print(len(value1_predictions))


    test_diff = delta_test_df['Value1'].tolist()
    print('Test diff shape: ', len(test_diff))

    # Calculate the R2 score using the predictions and the true values
    score = r2_score(test_diff, value1_predictions)

    # Plot the predictions and the true values
    plt.figure(figsize=(20,10))
    plt.plot(test_diff, label="true")
    plt.plot(value1_predictions, label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Returns ($)', fontsize=16)
    title_name = 'Predicting VAR model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/VAR-' + timeseries_name + '.png')
    # plt.show()
    # Save predictions into CSV
    predictions_df = pd.DataFrame(value1_predictions, columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/VAR-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/VAR-true_values.csv', index=False)
    return value1_predictions, test_diff, score