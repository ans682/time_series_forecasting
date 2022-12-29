from src.data_processing.load_bike import create_bike_df
from src.data_processing.load_stock import create_stock_df
from src.base_models.ar import ar_predict, ar_bike_predict
from src.base_models.arima import arima_predict, arima_bike_predict
from src.base_models.theta import theta_predict, theta_bike_predict
from src.base_models.rfr import rfr_predict, rfr_bike_predict
from src.base_models.lstm import lstm_predict, lstm_bike_predict
from src.base_models.var import var_predict, var_bike_predict
from src.ensemble_methods.rf_ensemble import rfr_ensemble_predict, rfr_ensemble_bike_predict
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import re

with open("data.yaml", "r") as stream:
    try:
        yaml_content = yaml.safe_load(stream)
        default_parameters = yaml_content['defaults']
        arima_parameters = default_parameters['arima']
        rfr_parameters = default_parameters['random_forest_regressor']
        var_parameters = default_parameters['var']
        theta_parameters = default_parameters['theta']
        sgdr_parameters = default_parameters['sgdr']
        lstm_parameters = default_parameters['lstm']
        input_path = yaml_content['input_path']
        output_path = yaml_content['output_path']
        input_file_name = yaml_content['input_file_name']
        time_series_name = yaml_content['time_series_name']
        bike_months_list = yaml_content['bike_months']
        bike_stations = yaml_content['bike_stations']

    except yaml.YAMLError as exc:
        print(exc)


call_package = True
is_one_base_model = False
is_financial_ts = False
is_ensemble = False
is_bike_ts_valid = False
user_base_models = []
r2_scores = []

print('Welcome! This package allows you to make time series forecasting by using six different base models and ensemble method.')
print('The package contains six base models:\n')
print('1. AutoRegression (AR)')
print('2. ARIMA')
print('3. Theta')
print('4. Vector AutoRegression (VAR)')
print('5. Random Forest Regressor (RFR)')
print('6. Long Short-Term Memory (LSTM)')
print('The package contains an ensemble Random Forest Regressor model which calculates the averages of the predictions from the selected base models \n')
print('If you want to change parameters of certain forecasting models or specify path to your time series, please edit "data.yaml" file. \n')

while call_package:
    command = input('Do you want to run experiment with one base model only? (yes or no). Press "q" to exit: ')
    if command.lower().split()[0][0] == 'y':
        is_one_base_model = True
        user_base_model_ids = input("Please list numeric ID of base model: ")
        clean_ids = re.sub('\s+', '', user_base_model_ids)
        user_base_models = [clean_ids.split()[0]]

    elif command.lower().split()[0][0] == 'n':
        # Ask user to add base models to the list
        user_base_model_ids = input("Please list numeric IDs of base models separated by comma: ")
        clean_ids = re.sub('\s+', '', user_base_model_ids)
        user_base_models = clean_ids.split(',')

        # Ask user if he or she wants to use ensemble method
        ensemble_choice = input('Include ensemble method? (yes or no):')
        if ensemble_choice.lower().split()[0][0] == 'y':
            is_ensemble = True
        
    
    elif command.lower().split()[0][0] == 'q':
        call_package = False
        break

    
    # Determine whether user's time series is financial or non-financial
    ts_type_command = input('Is your time series related to finance like stock prices? (yes or no): ')
    if ts_type_command.lower().split()[0][0] == 'y':
        is_financial_ts = True
        break
    elif ts_type_command.lower().split()[0][0] == 'n':
        is_financial_ts = False
        bike_file_type = input('Is your time series in the valid format? [Check README file] (yes or no):')
        if bike_file_type.lower().split()[0][0] == 'y':
            is_bike_ts_valid = True

        break
    elif command.lower().split()[0][0] == 'q':
        call_package = False
        break
    
print('User selections: ')
print('Is one base model: ', is_one_base_model)
print('Is financial ts: ', is_financial_ts)
print('Is ensemble: ', is_ensemble)
print('Is bike ts valid: ', is_bike_ts_valid)
print('List of base models: ', user_base_models)

base_models_list = []
for user_choice in user_base_models:
    if user_choice == '1':
        base_models_list.append('ar')
    elif user_choice == '2':
        base_models_list.append('arima')
    elif user_choice == '3':
        base_models_list.append('theta')
    elif user_choice == '4':
        base_models_list.append('var')
    elif user_choice == '5':
        base_models_list.append('rfr')
    elif user_choice == '6':
        base_models_list.append('lstm')

# If user works with financial time series,
if is_financial_ts:
    # do financial forecasting
    # Load dataset
    all_months_df = create_stock_df(input_path, input_file_name, output_path).dropna() # Create a df from input files and drop NaN values

    # Split into train and test datasets
    train, test = train_test_split(all_months_df, test_size=0.2, shuffle=False) # IF MULTIVARIATE TS
    print('Type of train: ', type((train)))
    print('Len of train: ', train.shape[0])
    print('Len of test: ', test.shape[0])

    # If ensemble method was selected, do the following
    base_model_predictions = {}

    if not is_one_base_model:
        for base_model in base_models_list:
            if base_model == 'ar':
                predictions, y_test, score = ar_predict(train, test, output_path, time_series_name, sgdr_parameters)
                print('AR1 r2-score: ', score)
                r2_scores.append(['AR', score])
                base_model_predictions['ar'] = predictions[1:]
            elif base_model == 'arima':
                predictions, y_test, score = arima_predict(train, test, output_path, time_series_name, arima_parameters)
                print('ARIMA r2-score: ', score)
                r2_scores.append(['ARIMA', score])
                base_model_predictions['arima'] = predictions[1:]
            elif base_model == 'theta':
                predictions, y_test, score = theta_predict(train, test, output_path, time_series_name, theta_parameters)
                print('Theta r2-score: ', score)
                r2_scores.append(['Theta', score])
                base_model_predictions['theta'] = predictions[1:]
            elif base_model == 'var':
                predictions, y_test, score = var_predict(train, test, output_path, time_series_name, var_parameters['maxlags'])
                print('VAR r2-score: ', score)
                r2_scores.append(['VAR', score])
                base_model_predictions['var'] = predictions[1:]
            elif base_model == 'rfr':
                predictions, y_test, score = rfr_predict(train, test, output_path, time_series_name, rfr_parameters)
                print('RFR r2-score: ', score)
                r2_scores.append(['RFR', score])
                base_model_predictions['rfr'] = predictions[1:]
            elif base_model == 'lstm':
                predictions, y_test, score = lstm_predict(train, test, output_path, time_series_name, lstm_parameters)
                print('LSTM r2-score: ', score)
                r2_scores.append(['LSTM', score])
                base_model_predictions['lstm'] = predictions[1:]
        
        if is_ensemble:
            # Call rfr_ensemble
            ensemble_predictions, y_test, ensemble_score = rfr_ensemble_predict(train, test, output_path, time_series_name, base_model_predictions)
            print('RF ensemble r2-score: ', ensemble_score)
            r2_scores.append(['RF ensemble', ensemble_score])

    
    # If user wants to run experiment with one base model only
    else:
        if base_models_list[0] == 'ar':
            predictions, y_test, score = ar_predict(train, test, output_path, time_series_name, sgdr_parameters)
            print('AR1 r2-score: ', score)
            r2_scores.append(['AR', score])
            base_model_predictions['ar'] = predictions
        elif base_models_list[0] == 'arima':
            predictions, y_test, score = arima_predict(train, test, output_path, time_series_name, arima_parameters)
            print('ARIMA r2-score: ', score)
            r2_scores.append(['ARIMA', score])
            base_model_predictions['arima'] = predictions
        elif base_models_list[0] == 'theta':
            predictions, y_test, score = theta_predict(train, test, output_path, time_series_name, theta_parameters)
            print('Theta r2-score: ', score)
            r2_scores.append(['Theta', score])
            base_model_predictions['theta'] = predictions
        elif base_models_list[0] == 'var':
            predictions, y_test, score = var_predict(train, test, output_path, time_series_name, var_parameters['maxlags'])
            print('VAR r2-score: ', score)
            r2_scores.append(['VAR', score])
            base_model_predictions['var'] = predictions
        elif base_models_list[0] == 'rfr':
            predictions, y_test, score = rfr_predict(train, test, output_path, time_series_name, rfr_parameters)
            print('RFR r2-score: ', score)
            r2_scores.append(['RFR', score])
            base_model_predictions['rfr'] = predictions
        elif base_models_list[0] == 'lstm':
            predictions, y_test, score = lstm_predict(train, test, output_path, time_series_name, lstm_parameters)
            print('LSTM r2-score: ', score)
            r2_scores.append(['LSTM', score])
            base_model_predictions['lstm'] = predictions
    


else:
    # do bike demand forecasting
    all_months_df = create_bike_df(input_path, bike_months_list, output_path, bike_stations, input_file_name, is_bike_ts_valid)  

    # Split into train and test datasets
    train, test = train_test_split(all_months_df, test_size=0.2, shuffle=False)
    print('Type of train: ', type((train)))
    print('Len of train: ', train.shape[0])
    print('Len of test: ', test.shape[0])


    # If ensemble method was selected, do the following
    base_model_predictions = {}

    if not is_one_base_model:
        for base_model in base_models_list:
            if base_model == 'ar':
                predictions, y_test, score = ar_bike_predict(train, test, output_path, time_series_name, sgdr_parameters)
                print('AR1 r2-score: ', score)
                r2_scores.append(['AR', score])
                base_model_predictions['ar'] = predictions
            elif base_model == 'arima':
                predictions, y_test, score = arima_bike_predict(train, test, output_path, time_series_name, arima_parameters)
                print('ARIMA r2-score: ', score)
                r2_scores.append(['ARIMA', score])
                base_model_predictions['arima'] = predictions
            elif base_model == 'theta':
                predictions, y_test, score = theta_bike_predict(train, test, output_path, time_series_name, theta_parameters)
                print('Theta r2-score: ', score)
                r2_scores.append(['Theta', score])
                base_model_predictions['theta'] = predictions
            elif base_model == 'var':
                predictions, y_test, score = var_bike_predict(train, test, output_path, time_series_name, var_parameters['maxlags'])
                print('VAR r2-score: ', score)
                r2_scores.append(['VAR', score])
                base_model_predictions['var'] = predictions
            elif base_model == 'rfr':
                predictions, y_test, score = rfr_bike_predict(train, test, output_path, time_series_name, rfr_parameters)
                print('RFR r2-score: ', score)
                r2_scores.append(['RFR', score])
                base_model_predictions['rfr'] = predictions
            elif base_model == 'lstm':
                predictions, y_test, score = lstm_bike_predict(train, test, output_path, time_series_name, lstm_parameters)
                print('LSTM r2-score: ', score)
                r2_scores.append(['LSTM', score])
                base_model_predictions['lstm'] = predictions
        
        if is_ensemble:
            # Call rfr_ensemble
            ensemble_predictions, y_test, ensemble_score = rfr_ensemble_bike_predict(train, test, output_path, time_series_name, base_model_predictions)
            print('RF ensemble r2-score: ', ensemble_score)
            r2_scores.append(['RF ensemble', ensemble_score])
    
    # If user wants to run experiment with one base model only
    else:
        if base_models_list[0] == 'ar':
            predictions, y_test, score = ar_bike_predict(train, test, output_path, time_series_name, sgdr_parameters)
            print('AR1 r2-score: ', score)
            r2_scores.append(['AR', score])
            base_model_predictions['ar'] = predictions
        elif base_models_list[0] == 'arima':
            predictions, y_test, score = arima_bike_predict(train, test, output_path, time_series_name, arima_parameters)
            print('ARIMA r2-score: ', score)
            r2_scores.append(['ARIMA', score])
            base_model_predictions['arima'] = predictions
        elif base_models_list[0] == 'theta':
            predictions, y_test, score = theta_bike_predict(train, test, output_path, time_series_name, theta_parameters)
            print('Theta r2-score: ', score)
            r2_scores.append(['Theta', score])
            base_model_predictions['theta'] = predictions
        elif base_models_list[0] == 'var':
            predictions, y_test, score = var_bike_predict(train, test, output_path, time_series_name, var_parameters['maxlags'])
            print('VAR r2-score: ', score)
            r2_scores.append(['VAR', score])
            base_model_predictions['var'] = predictions
        elif base_models_list[0] == 'rfr':
            predictions, y_test, score = rfr_bike_predict(train, test, output_path, time_series_name, rfr_parameters)
            print('RFR r2-score: ', score)
            r2_scores.append(['RFR', score])
            base_model_predictions['rfr'] = predictions
        elif base_models_list[0] == 'lstm':
            predictions, y_test, score = lstm_bike_predict(train, test, output_path, time_series_name, lstm_parameters)
            print('LSTM r2-score: ', score)
            r2_scores.append(['LSTM', score])
            base_model_predictions['lstm'] = predictions


# Save r2 scores into CSV
r2_scores_df = pd.DataFrame(r2_scores, columns=['Forecasting method','R2-score'])
r2_scores_df.to_csv(output_path + '/r2-scores.csv', index=False)
