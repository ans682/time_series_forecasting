import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from ..helpers.helpers import difference
import time
import pandas as pd
import numpy as np
##################################################################################################################################

##################################################################################################################################
def ar_bike_predict(train, test, output_plots_path, timeseries_name, sgdr_parameters):
    train_raw = train
    test_raw = test

    train_size = train.shape[0] - 1
    test_size = test.shape[0]
    num_features = sgdr_parameters['num_features']

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
    
    # Number of subarrays
    total_num_subarrays = total_data_size - num_features + 1
    train_num_subarrays = train_size - num_features + 1
    test_num_subarrays = total_num_subarrays - train_num_subarrays

    # Create lists to store x_features and y_features
    total_x_features = []
    total_y_features = total_data[num_features - 1:]
    
    # Fill out x_features
    for i in range(total_num_subarrays): # 
        current_i = i + num_features # 
        x_feature = []
        for j in range(num_features):
            x_feature.append(total_data[current_i - j - 1])
        
        # Append x_feature to total_x_features list
        total_x_features.append(x_feature[::-1]) # REVERSE LIST

    base_training_X = total_x_features[:train_num_subarrays]
    base_training_y = total_y_features[:train_num_subarrays]
    print(len(base_training_X) == len(base_training_y))

    start_time = time.time()

    # Create an object of SGDRegressor
    ar1 = SGDRegressor(shuffle=False, learning_rate=sgdr_parameters['learning_rate'])
    ar1.fit(base_training_X, base_training_y)
    score = ar1.score(base_training_X, base_training_y)
    # print("AR r2-score = ", score)
    predictions = []

    # Loop through test data:
    for i in range(test_num_subarrays): # [10,11,12,13]
        current_i = i + train_num_subarrays
        prediction = ar1.predict([total_x_features[current_i]])
        predictions.append(prediction)
        # Retrain AR model
        # ar1.partial_fit([total_data[current_i]], [total_y_features[current_i]])
        ar1.fit([total_x_features[current_i]], [total_y_features[current_i]])

    end_time = time.time()
    total_time = end_time - start_time
    print("Time executed: ", total_time)
    
    print('Size of predictions: ', len(predictions))
    print('Size of test data: ', test.shape[0])
    
    test_diff = test_raw['Value1'].tolist()

    new_predictions = []
    for pred in predictions:
        new_predictions.append(pred[0])
    predictions = new_predictions
    print('Size of reshaped AR predictions', len(predictions))
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
    score = r2_score(test_diff, predictions[1:])

    # Plot the predictions and the true values
    plt.figure(figsize=(20,10), dpi=300)
    plt.plot(test_diff, label="true")
    plt.plot(predictions[1:], label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Number of bikes demanded', fontsize=16)
    title_name = 'Predicting AR1 model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/AR1-' + timeseries_name + '.png')
    # plt.show()
    # Save predictions into CSV
    predictions_df = pd.DataFrame(predictions[1:], columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/AR1-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/AR1-true_values.csv', index=False)

    return predictions[1:], test_diff, score
##################################################################################################################################

##################################################################################################################################
# Financial time series predictions
def ar_predict(train, test, output_plots_path, timeseries_name, sgdr_parameters):
    train_raw = train
    test_raw = test

    train_size = train.shape[0] - 1
    test_size = test.shape[0]
    num_features = sgdr_parameters['num_features']

    total_data = pd.concat([train, test], axis=0)
    total_data = total_data['Value1'].tolist()

    # Join train and test
    # total_data = train.tolist() + test.tolist() # This is the merged list now
    
    # Difference time series, i.e. use returns
    total_data = difference(total_data)
    total_size = len(total_data)
    print('Size of differenced total data: ', len(total_data))
    
    # Number of subarrays
    total_num_subarrays = total_size - num_features + 1
    train_num_subarrays = train_size - num_features + 1
    test_num_subarrays = total_num_subarrays - train_num_subarrays

    # Create lists to store x_features and y_features
    total_x_features = []
    total_y_features = total_data[num_features - 1:]
    
    # Fill out x_features
    for i in range(total_num_subarrays): # 12 subarrs
        current_i = i + num_features # 
        x_feature = []
        for j in range(num_features):
            x_feature.append(total_data[current_i - j - 1])
        
        # Append x_feature to total_x_features list
        total_x_features.append(x_feature[::-1]) # REVERSE LIST

    base_training_X = total_x_features[:train_num_subarrays]
    base_training_y = total_y_features[:train_num_subarrays]
    print(len(base_training_X) == len(base_training_y))

    start_time = time.time()

    # Create an object of SGDRegressor
    ar1 = SGDRegressor(shuffle=False, learning_rate=sgdr_parameters['learning_rate'])
    ar1.fit(base_training_X, base_training_y)
    score = ar1.score(base_training_X, base_training_y)
    # print("AR r2-score = ", score)
    predictions = []

    # Loop through test data:
    for i in range(test_num_subarrays): # [10,11,12,13]
        current_i = i + train_num_subarrays
        prediction = ar1.predict([total_x_features[current_i]])
        predictions.append(prediction)
        # Retrain AR model
        # ar1.partial_fit([total_data[current_i]], [total_y_features[current_i]])
        ar1.fit([total_x_features[current_i]], [total_y_features[current_i]])

    end_time = time.time()
    total_time = end_time - start_time
    print("Time executed: ", total_time)
    
    print('Size of predictions: ', len(predictions))
    print('Size of test data: ', test.shape[0])
    
    test_diff = difference(test_raw['Value1'].tolist())

    new_predictions = []
    for pred in predictions:
        new_predictions.append(pred[0])
    predictions = new_predictions
    print('Size of reshaped AR predictions', len(predictions))

    # Calculate the R2 score using the predictions and the true values
    score = r2_score(test_diff, predictions[1:])


    # Plot the predictions and the true values
    plt.figure(figsize=(20,10))
    plt.plot(test_diff, label="true")
    plt.plot(predictions[1:], label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Returns ($)', fontsize=16)
    title_name = 'Predicting AR1 model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/AR1-' + timeseries_name + '.png')
    # plt.show()

    # Save predictions into CSV
    predictions_df = pd.DataFrame(predictions[1:], columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/AR1-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/AR1-true_values.csv', index=False)

    return predictions, test_diff, score