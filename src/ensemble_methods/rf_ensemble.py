import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from ..helpers.helpers import difference
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import copy

##################################################################################################################################
#RFR Meta Model
def rfr_ensemble_bike_predict(train, test, output_plots_path, timeseries_name, base_model_predictions):
    num_base_models = len(base_model_predictions)

    train_raw = train
    test_raw = test
    test_diff = test_raw['Value1'].tolist()

    train_size = train.shape[0]
    test_size = test.shape[0]
    num_features = 25

    # Convert base_model_predictions dict into df
    base_models_df = pd.DataFrame(base_model_predictions) 
    base_models_df['Test'] = test_diff

    base_models_df_raw = copy.deepcopy(base_models_df)
    
    # Save predictions into csv
    # base_models_df.to_csv(input_path + '/base_model_predictions.csv')

    base_model_names = base_models_df.columns.tolist()

    total_data = pd.concat([train, test], axis=0)

    total_data_size = total_data.shape[0]

    ######### SECTION FOR RESCALING THE TOTAL DATASET ###########   
    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    base_models_df[base_model_names] = scaler.fit_transform(base_models_df[base_model_names])
    ##############################################################
    
    predictions_meta_rfr = []
    meta_model_rfr = RandomForestRegressor(bootstrap=False)

    
    new_train_size = int(base_models_df.shape[0] * 0.8)
    print('New Train Size: ', new_train_size)
    new_test_size = base_models_df.shape[0] - new_train_size

    base_models_df_rescale_input = base_models_df_raw[:new_test_size]


    meta_X = []
    for i in range(new_train_size):
        meta_X_instance = []
        for b_m in range(num_base_models):
            meta_X_instance.append(base_models_df.iloc[i][b_m])

        meta_X.append(meta_X_instance)
    
    start_time = time.time()
    # Fit Meta RFR Ensemble
    meta_model_rfr.fit(meta_X, base_models_df['Test'][:new_train_size])

    
    for t_i in (range(new_test_size)):
        current_t = t_i + new_train_size

        meta_model_rfr.fit(meta_X, base_models_df['Test'][:current_t])
        # Update meta_X
        meta_instance = []
        for b_m in range(num_base_models):
            meta_instance.append(base_models_df.iloc[current_t][b_m])
        meta_X.append(meta_instance)
        print('SHAPE of New META X: ', len(meta_X))

        meta_current_instance = []
        for b_m in range(num_base_models):
            meta_current_instance.append(base_models_df.iloc[current_t][b_m])
        prediction = meta_model_rfr.predict([meta_current_instance])
        predictions_meta_rfr.append(prediction)
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Time executed: ", total_time)
    print("Size of META RFR predictions: ", len(predictions_meta_rfr))
    print('Size of Test: ', len(test_diff))
    
    # Inverse transform the predictions
############################################################
    # Convert predictions from list into numpy array. Then reshape into 2d array.
    predictions_size = len(predictions_meta_rfr)
    predictions_meta_rfr = np.array(predictions_meta_rfr).reshape(predictions_size, 1)

    # Convert predictions to list
    predictions_meta_rfr = predictions_meta_rfr.reshape(-1).tolist()

    base_models_df_rescale_input['Test'] = predictions_meta_rfr
    # Inverse transform predictions
    transformed_df = scaler.inverse_transform(base_models_df_rescale_input)
    # predictions_meta_rfr = transformed_df.iloc[:,-1:].tolist()
    predictions = []
    for pred in transformed_df:
        predictions.append(pred[-1])
    predictions_meta_rfr = predictions
    
############################################################
    test_diff = test_diff[-new_test_size:]
    score = r2_score(test_diff, predictions_meta_rfr)

    # Plot the predictions and the true values
    plt.figure(figsize=(20,10),dpi=300)
    plt.plot(test_diff, label="true")
    plt.plot(predictions_meta_rfr, label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Number of bikes demanded', fontsize=16)
    title_name = 'Predicting Ensemble Random Forest Regressor model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/Ensemble_RFR-' + timeseries_name + '.png')
    # plt.show()
    # Save predictions into CSV
    predictions_df = pd.DataFrame(predictions_meta_rfr, columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/Ensemble_RFR-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/Ensemble_RFR-true_values.csv', index=False)

    return predictions_meta_rfr, test_diff, score


##################################################################################################################################

##################################################################################################################################
#RFR Meta Model
def rfr_ensemble_predict(train, test, output_plots_path, timeseries_name, base_model_predictions):
    num_base_models = len(base_model_predictions)

    train_raw = train
    test_raw = test
    test_diff = difference(test_raw['Value1'].tolist())

    train_size = train.shape[0] - 1
    test_size = test.shape[0]
    num_features = 25

    ''''''
    # Convert base_model_predictions dict into df
    base_models_df = pd.DataFrame(base_model_predictions) 
    base_models_df['Test'] = test_diff

    base_models_df_raw = copy.deepcopy(base_models_df)
    
    # Save predictions into csv
    # base_models_df.to_csv(input_path + '/base_model_predictions.csv')

    base_model_names = base_models_df.columns.tolist()

    ''''''

    total_data = pd.concat([train, test], axis=0)
    
    total_data_size = total_data.shape[0]
    total_data = total_data['Value1'].tolist()
    
    # Difference time series, i.e. use returns
    total_data = difference(total_data)
    total_size = len(total_data)
    print('Size of differenced total data: ', len(total_data))
    
    predictions_meta_rfr = []
    meta_model_rfr = RandomForestRegressor(bootstrap=False)

    
    # new_train_size = int(len(base_model_predictions[0]) * 0.8)
    new_train_size = int(base_models_df.shape[0] * 0.8)
    print('New Train Size: ', new_train_size)
    new_test_size = base_models_df.shape[0] - new_train_size


    meta_X = []
    for i in range(new_train_size):
        # Loop through all base_models and append their prediction to a current meta_X_instance
        meta_X_instance = []
        for b_m in range(num_base_models):
            meta_X_instance.append(base_models_df.iloc[i][b_m])

        # meta_X_instance = [base_models_df.iloc[i][0], base_models_df.iloc[i][1], base_models_df.iloc[i][2], base_models_df.iloc[i][3]]
        # meta_X_instance = [base_model_predictions[0][i], base_model_predictions[1][i], base_model_predictions[2][i], base_model_predictions[3][i]]
        meta_X.append(meta_X_instance)
    
    start_time = time.time()
    # Fit Meta RFR Ensemble
    meta_model_rfr.fit(meta_X, test_diff[:new_train_size])

    
    for t_i in (range(new_test_size)):
        current_t = t_i + new_train_size

        meta_model_rfr.fit(meta_X, test_diff[:current_t])
        # Update meta_X
        meta_instance = []
        for b_m in range(num_base_models):
            meta_instance.append(base_models_df.iloc[current_t][b_m])

        # meta_instance = [base_model_predictions[0][current_t], base_model_predictions[1][current_t], base_model_predictions[2][current_t], base_model_predictions[3][current_t]]
        meta_X.append(meta_instance)
        print('SHAPE of New META X: ', len(meta_X))
        print('Size of NEW test_diff: ', len(test_diff[:current_t + 1]))

        meta_current_instance = []
        for b_m in range(num_base_models):
            meta_current_instance.append(base_models_df.iloc[current_t][b_m])

        prediction = meta_model_rfr.predict([meta_current_instance])
        # prediction = meta_model_rfr.predict([[base_model_predictions[0][current_t], base_model_predictions[1][current_t], base_model_predictions[2][current_t], base_model_predictions[3][current_t]]])
        predictions_meta_rfr.append(prediction)
        # meta_X.append([all_predictions[0][current_t][0], all_predictions[1][current_t], all_predictions[2][current_t][0], all_predictions[3][current_t]])
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Time executed: ", total_time)
    print("Size of META RFR predictions: ", len(predictions_meta_rfr))
    print('Size of Test: ', len(test_diff))
    
    # Calculate the R2 score using the predictions and the true values
    test_diff = test_diff[-new_test_size:]
    score = r2_score(test_diff, predictions_meta_rfr)

    # Plot the predictions and the true values
    plt.figure(figsize=(20,10))
    plt.plot(test_diff, label="true")
    plt.plot(predictions_meta_rfr, label="predicted")
    plt.legend()
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Returns ($)', fontsize=16)
    title_name = 'Predicting Ensemble Random Forest Regressor model for ' + timeseries_name
    plt.title(title_name, fontsize=20)
    
    plt.savefig(output_plots_path + '/Ensemble_RFR-' + timeseries_name + '.png')
    # plt.show()
    # Save predictions into CSV
    predictions_df = pd.DataFrame(predictions_meta_rfr, columns=['Predictions'])
    predictions_df.to_csv(output_plots_path + '/Ensemble_RFR-predictions.csv', index=False)

    # Save ground truth values into CSV
    true_values_df = pd.DataFrame(test_diff, columns=['True_values'])
    true_values_df.to_csv(output_plots_path + '/Ensemble_RFR-true_values.csv', index=False)
    
    return predictions_meta_rfr, test_diff, score