from statsmodels.tsa.forecasting.theta import ThetaModel
import pandas as pd
from ..helpers.helpers import difference
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
##################################################################################################################################

##################################################################################################################################
def theta_bike_predict(train, test, output_plots_path, timeseries_name, theta_parameters):
    train_raw = train
    test_raw = test

    train_size = train.shape[0] - 1
    test_size = test.shape[0]
    # num_features = 25

    total_data = pd.concat([train, test], axis=0)
    ######### SECTION FOR RESCALING THE TOTAL DATASET ###########
    total_data_size = total_data.shape[0]
    print('Size of differenced total data: ', len(total_data))
    print('Type of total data: ', type(total_data))

    # Get 'Value1' column's values
    total_data = total_data['Value1'].values.reshape(total_data_size, 1)

    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit data into scaler
    scaler = scaler.fit(total_data)

    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    # Transform the dataset 
    total_data = scaler.transform(total_data).reshape(-1).tolist()
    ##############################################################
    
    predictions = []
    
    # Convert total_data list to pd.Series
    total_data = pd.Series(total_data)
    start_time = time.time()

    for t_i in (range(test_size)):
        if t_i % 50 == 0:
            print('Completed ', t_i, ' predictions')
        current_t = t_i + train_size
        #######
        model = ThetaModel(total_data[:current_t],deseasonalize=theta_parameters['deseasonalize'])
        fitted_model = model.fit()
        prediction = fitted_model.forecast().reset_index(drop=True)
        predictions.append(prediction[0])
        #######
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Time executed: ", total_time)
    print("Size of predictions: ", len(predictions))

    test_diff = test_raw['Value1'].tolist()

############################################################
    # Convert predictions from list into numpy array. Then reshape into 2d array.
    predictions_size = len(predictions)
    predictions = np.array(predictions).reshape(predictions_size, 1)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)

    # Convert predictions to list
    predictions = predictions.reshape(-1).tolist()
############################################################

    # Calculate the R2 score using the predictions and the true values
    score = r2_score(test_diff, predictions)

    # Plot the predictions and the true values
    plt.figure(figsize=(20,10))
    plt.plot(test_diff, label="true")
    plt.plot(predictions, label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Number of bikes demanded', fontsize=16)
    title_name = 'Predicting Theta model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/Theta-' + timeseries_name + '.png')
    # plt.show()

    # Save predictions into CSV
    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/Theta-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/Theta-true_values.csv', index=False)

    return predictions, test_diff, score

##################################################################################################################################

##################################################################################################################################
def theta_predict(train, test, output_plots_path, timeseries_name, theta_parameters):
    train_raw = train
    test_raw = test

    train_size = train.shape[0] - 1
    test_size = test.shape[0]
    # num_features = 25

    total_data = pd.concat([train, test], axis=0)
    total_data = total_data['Value1'].tolist()

    # # Join train and test
    # total_data = train.tolist() + test.tolist() # This is the merged list now
    
    # Difference time series, i.e. use returns
    total_data = difference(total_data)
    total_size = len(total_data)
    print('Size of differenced total data: ', len(total_data))
    
    predictions = []
    
    # Convert total_data list to pd.Series
    total_data = pd.Series(total_data)
    start_time = time.time()

    for t_i in (range(test_size)):
        if t_i % 50 == 0:
            print('Completed ', t_i, ' predictions')
        current_t = t_i + train_size
        #######
        model = ThetaModel(total_data[:current_t],deseasonalize=theta_parameters['deseasonalize'])
        fitted_model = model.fit()
        prediction = fitted_model.forecast().reset_index(drop=True)
        predictions.append(prediction[0])
        #######
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Time executed: ", total_time)
    print("Size of predictions: ", len(predictions))

    test_diff = difference(test_raw['Value1'].tolist())

    # Calculate the R2 score using the predictions and the true values
    score = r2_score(test_diff, predictions[1:])

    # Plot the predictions and the true values
    plt.figure(figsize=(20,10))
    plt.plot(test_diff, label="true")
    plt.plot(predictions[1:], label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Returns ($)', fontsize=16)
    title_name = 'Predicting Theta model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/Theta-' + timeseries_name + '.png')
    # plt.show()
    # Save predictions into CSV
    predictions_df = pd.DataFrame(predictions[1:], columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/Theta-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/Theta-true_values.csv', index=False)
    return predictions, test_diff, score