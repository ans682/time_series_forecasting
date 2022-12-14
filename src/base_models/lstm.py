import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from ..helpers.helpers import difference
import time
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler

##################################################################################################################################

##################################################################################################################################
def lstm_bike_predict(train, test, output_plots_path, timeseries_name):
    train_raw = train
    test_raw = test
    # Save differenced test data for future references
    # test_diff = difference(test.tolist())

    train_size = train.shape[0] - 1
    test_size = test.shape[0]
    # num_features = 25

    # Join train and test
    # total_data = train.tolist() + test.tolist() # This is the merged list now
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


    # # Difference time series, i.e. use returns
    # total_data = difference(total_data)
    # total_size = len(total_data)
    # print('Size of differenced total data: ', len(total_data))

    # Convert to numpy array
    total_data = np.array(total_data)
    full_data = total_data.reshape(total_data.shape[0],-1)

    # Choosing between Standardization or normalization
    #sc = StandardScaler()
    sc=MinMaxScaler()

    DataScaler = sc.fit(full_data)
    X=DataScaler.transform(full_data)

    # Split into X and y samples
    X_samples = []
    y_samples = []

    num_rows = len(X)
    time_steps=10  # next day's Price Prediction is based on last how many past day's prices
    num_units = 1

    # Update train_size based on the num of time_steps
    train_size -= time_steps

    # Iterate thru the values to create combinations
    for i in range(time_steps , num_rows , num_units):
        x_sample = X[i  - time_steps:i]
        y_sample = X[i]
        X_samples.append(x_sample)
        y_samples.append(y_sample)

    ################################################
    # Reshape the Input as a 3D (number of samples, time steps, features)
    X_data=np.array(X_samples)
    X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
    print('\n#### Input Data shape ####')
    print(X_data.shape)

    # We do not reshape y as a 3D data  as it is supposed to be a single column only
    y_data=np.array(y_samples)
    y_data=y_data.reshape(y_data.shape[0], 1)
    print('\n#### Output Data shape ####')
    print(y_data.shape)
        
    # Train and test data
    # Splitting the data into train and test
    X_train=X_data[:train_size]
    X_test=X_data[train_size:]

    y_train=y_data[:train_size]
    y_test=y_data[train_size:]
    
    ############################################
    
    # Printing the shape of training and testing
    print('\n#### Training Data shape ####')
    print(X_train.shape)
    print(y_train.shape)
    print('\n#### Testing Data shape ####')
    print(X_test.shape)
    print(y_test.shape)
    
    # Defining Input shapes for LSTM
    time_steps=X_train.shape[1]
    total_features=X_train.shape[2]
    print("Number of TimeSteps:", time_steps)
    print("Number of Features:", total_features)


    # Initialising the RNN
    regressor = Sequential()

    # Adding the First input hidden layer and the LSTM layer
    # return_sequences = True, means the output of every time step to be shared with hidden next layer
    regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (time_steps, total_features), return_sequences=True))

    # Adding the Second Second hidden layer and the LSTM layer
    regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (time_steps, total_features), return_sequences=True))

    # Adding the Second Third hidden layer and the LSTM layer
    regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    ##################################################
    # print('X test: ', X_test)
    # print('Type of X test: ', type(X_test))

    # Measuring the time taken by the model to train
    predictions = []
    StartTime=time.time()
    for i in range(test_size):
        if i % 10 == 0: ##### CHANGE THIS TO 25
            # Fitting the RNN to the Training set
            seen_X_test = X_test[:i].reshape(i, time_steps, 1)
            print('seen X_test shape: ', seen_X_test.shape)

            input_X = np.concatenate((X_train, seen_X_test), axis=0)
            input_y = np.concatenate((y_train, y_test[:i]), axis=0)

            regressor.fit(input_X, input_y, batch_size = 5, epochs = 100)
            print('Completed ', i, ' predictions...')
        
        # Making predictions on test data
        # print('printing X-test[i] ...')
        # print(X_test[i])
        # print('Shape of X_test[i]: ', X_test[i].shape)
        cur_x = X_test[i].reshape(1, time_steps, 1)
        print('Shape of cur x: ', cur_x.shape)
        predicted_Price = regressor.predict(cur_x)
        # print('After reshaping: ', cur_x.shape)


        predicted_Price = DataScaler.inverse_transform(predicted_Price)
        predictions.append(predicted_Price)

    EndTime=time.time()
    total_time = EndTime - StartTime
    print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')
    print('Time executed: ', total_time)
    ##########################################################
    print('PREDICTIONS: ', predictions)
    print('/////////////')
    # Measuring the accuracy of the model on test data
    # Making predictions on test data
    ### predicted_Price = regressor.predict(X_test)
    ### predicted_Price = DataScaler.inverse_transform(predicted_Price)

    # Getting the original price values for testing data
    orig=y_test
    orig=DataScaler.inverse_transform(y_test)

    # Accuracy of the predictions
    # print('Accuracy:', 100 - (100*(abs(orig-predicted_Price)/orig)).mean())

    orig = orig.reshape(-1).tolist()
    predictions_clean = []
    for pred in predictions:
        predictions_clean.append(pred[0][0])
    # predictions = predictions.reshape(-1).tolist()
    print('Clean predictions: ', predictions_clean)
    # predicted_Price = predicted_Price.reshape(-1).tolist()
    # Calculate the R2 score using the predictions and the true values
    score = r2_score(orig, predictions_clean)


    # Visualize the results
    plt.figure(figsize=(20,10))
    plt.plot(orig, label="true")
    plt.plot(predictions_clean, label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Returns ($)', fontsize=16)
    title_name = 'Predicting LSTM model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/LSTM-' + timeseries_name + '.png')

    # predictions = []
    return predictions_clean, orig, score

##################################################################################################################################

##################################################################################################################################
def lstm_predict(train, test, output_plots_path, timeseries_name):
    train_raw = train
    test_raw = test
    # Save differenced test data for future references
    test_diff = difference(test.tolist())

    train_size = train.shape[0] - 1
    test_size = test.shape[0]
    # num_features = 25

    # Join train and test
    total_data = train.tolist() + test.tolist() # This is the merged list now

    # Difference time series, i.e. use returns
    total_data = difference(total_data)
    total_size = len(total_data)
    print('Size of differenced total data: ', len(total_data))

    # Convert to numpy array
    total_data = np.array(total_data)
    full_data = total_data.reshape(total_data.shape[0],-1)

    # Choosing between Standardization or normalization
    #sc = StandardScaler()
    sc=MinMaxScaler()

    DataScaler = sc.fit(full_data)
    X=DataScaler.transform(full_data)

    # Split into X and y samples
    X_samples = []
    y_samples = []

    num_rows = len(X)
    time_steps=10  # next day's Price Prediction is based on last how many past day's prices
    num_units = 1

    # Update train_size based on the num of time_steps
    train_size -= time_steps

    # Iterate thru the values to create combinations
    for i in range(time_steps , num_rows , num_units):
        x_sample = X[i  - time_steps:i]
        y_sample = X[i]
        X_samples.append(x_sample)
        y_samples.append(y_sample)

    ################################################
    # Reshape the Input as a 3D (number of samples, time steps, features)
    X_data=np.array(X_samples)
    X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
    print('\n#### Input Data shape ####')
    print(X_data.shape)

    # We do not reshape y as a 3D data  as it is supposed to be a single column only
    y_data=np.array(y_samples)
    y_data=y_data.reshape(y_data.shape[0], 1)
    print('\n#### Output Data shape ####')
    print(y_data.shape)
        
    # Train and test data
    # Splitting the data into train and test
    X_train=X_data[:train_size]
    X_test=X_data[train_size:]

    y_train=y_data[:train_size]
    y_test=y_data[train_size:]
    
    ############################################
    
    # Printing the shape of training and testing
    print('\n#### Training Data shape ####')
    print(X_train.shape)
    print(y_train.shape)
    print('\n#### Testing Data shape ####')
    print(X_test.shape)
    print(y_test.shape)
    
    # Defining Input shapes for LSTM
    time_steps=X_train.shape[1]
    total_features=X_train.shape[2]
    print("Number of TimeSteps:", time_steps)
    print("Number of Features:", total_features)


    # Initialising the RNN
    regressor = Sequential()

    # Adding the First input hidden layer and the LSTM layer
    # return_sequences = True, means the output of every time step to be shared with hidden next layer
    regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (time_steps, total_features), return_sequences=True))

    # Adding the Second Second hidden layer and the LSTM layer
    regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (time_steps, total_features), return_sequences=True))

    # Adding the Second Third hidden layer and the LSTM layer
    regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    ##################################################
    # print('X test: ', X_test)
    # print('Type of X test: ', type(X_test))

    # Measuring the time taken by the model to train
    predictions = []
    StartTime=time.time()
    for i in range(test_size):
        if i % 10 == 0: ##### CHANGE THIS TO 25
            # Fitting the RNN to the Training set
            seen_X_test = X_test[:i].reshape(i, time_steps, 1)
            print('seen X_test shape: ', seen_X_test.shape)

            input_X = np.concatenate((X_train, seen_X_test), axis=0)
            input_y = np.concatenate((y_train, y_test[:i]), axis=0)

            regressor.fit(input_X, input_y, batch_size = 5, epochs = 100)
            print('Completed ', i, ' predictions...')
        
        # Making predictions on test data
        # print('printing X-test[i] ...')
        # print(X_test[i])
        # print('Shape of X_test[i]: ', X_test[i].shape)
        cur_x = X_test[i].reshape(1, time_steps, 1)
        print('Shape of cur x: ', cur_x.shape)
        predicted_Price = regressor.predict(cur_x)
        # print('After reshaping: ', cur_x.shape)


        predicted_Price = DataScaler.inverse_transform(predicted_Price)
        predictions.append(predicted_Price)

    EndTime=time.time()
    total_time = EndTime - StartTime
    print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')
    print('Time executed: ', total_time)
    ##########################################################
    print('PREDICTIONS: ', predictions)
    print('/////////////')
    # Measuring the accuracy of the model on test data
    # Making predictions on test data
    ### predicted_Price = regressor.predict(X_test)
    ### predicted_Price = DataScaler.inverse_transform(predicted_Price)

    # Getting the original price values for testing data
    orig=y_test
    orig=DataScaler.inverse_transform(y_test)

    # Accuracy of the predictions
    # print('Accuracy:', 100 - (100*(abs(orig-predicted_Price)/orig)).mean())

    orig = orig.reshape(-1).tolist()
    predictions_clean = []
    for pred in predictions:
        predictions_clean.append(pred[0][0])
    # predictions = predictions.reshape(-1).tolist()
    print('Clean predictions: ', predictions_clean)
    # predicted_Price = predicted_Price.reshape(-1).tolist()
    # Calculate the R2 score using the predictions and the true values
    score = r2_score(orig, predictions_clean)


    # Visualize the results
    plt.figure(figsize=(20,10))
    plt.plot(orig, label="true")
    plt.plot(predictions_clean, label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Returns ($)', fontsize=16)
    title_name = 'Predicting LSTM model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/LSTM-' + timeseries_name + '.png')

    # predictions = []
    return predictions_clean, orig, score

