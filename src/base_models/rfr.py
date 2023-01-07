import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from ..helpers.helpers import difference
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import time
##################################################################################################################################

##################################################################################################################################
def rfr_bike_predict(train, test, output_plots_path, timeseries_name, rfr_parameters):
    train_raw = train
    test_raw = test

    train_size = train.shape[0] - 1
    test_size = test.shape[0]
    num_features = 25

    # Join train and test
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

    
    predictions = []

    start_time = time.time()
    rfr = RandomForestRegressor(bootstrap=rfr_parameters['bootstrap'])
    rfr.fit(base_training_X, base_training_y)


    for t_i in (range(test_size)):
        if t_i % 50 == 0:
            print('Completed ', t_i, ' predictions')
        current_t = t_i + train_num_subarrays
        # Fit Random Forest Regressor model
        rfr.fit(total_x_features[:current_t], total_y_features[:current_t])
        prediction = rfr.predict([total_x_features[current_t]])
        predictions.append(prediction[0])
    
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
    plt.figure(figsize=(20,10), dpi=300)
    plt.plot(test_diff, label="true")
    plt.plot(predictions, label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Number of bikes demanded', fontsize=16)
    title_name = 'Predicting Random Forest Regressor model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/RFR-' + timeseries_name + '.png')
    # plt.show()
    # Save predictions into CSV
    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/RFR-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/RFR-true_values.csv', index=False)

    return predictions, test_diff, score


##################################################################################################################################

##################################################################################################################################
def rfr_predict(train, test, output_plots_path, timeseries_name, rfr_parameters):
    train_raw = train
    # test_raw = test
    

    train_size = train.shape[0] - 1
    test_size = test.shape[0]
    num_features = 25
    # ///
    # Join train and test
    # total_data = train.tolist() + test.tolist() # This is the merged list now
    test_raw = test['Value1'].values.reshape(test_size, 1)
    test_raw = test_raw.reshape(-1).tolist()
    total_data = pd.concat([train, test], axis=0)

    ######### SECTION FOR RESCALING THE TOTAL DATASET ###########
    total_data_size = total_data.shape[0]
    print('Size of differenced total data: ', len(total_data))
    print('Type of total data: ', type(total_data))

    # Get 'Value1' column's values
    total_data = total_data['Value1'].values.reshape(total_data_size, 1)
    print('total_data: ', total_data.shape)
    total_data = total_data.reshape(-1).tolist()
    
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

    
    predictions = []

    start_time = time.time()
    rfr = RandomForestRegressor(bootstrap=rfr_parameters['bootstrap'])
    rfr.fit(base_training_X, base_training_y)


    for t_i in (range(test_size)):
        if t_i % 50 == 0:
            print('Completed ', t_i, ' predictions')
        current_t = t_i + train_num_subarrays
        # Fit Random Forest Regressor model
        rfr.fit(total_x_features[:current_t], total_y_features[:current_t])
        prediction = rfr.predict([total_x_features[current_t]])
        predictions.append(prediction[0])
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Time executed: ", total_time)
    print("Size of predictions: ", len(predictions))

    test_diff = difference(test_raw)

    # Calculate the R2 score using the predictions and the true values
    score = r2_score(test_diff, predictions[1:])

    # Plot the predictions and the true values
    plt.figure(figsize=(20,10))
    plt.plot(test_diff, label="true")
    plt.plot(predictions[1:], label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Returns ($)', fontsize=16)
    title_name = 'Predicting Random Forest Regressor model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/RFR-' + timeseries_name + '.png')
    # plt.show()
    # Save predictions into CSV
    predictions_df = pd.DataFrame(predictions[1:], columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/RFR-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/RFR-true_values.csv', index=False)

    return predictions, test_diff, score