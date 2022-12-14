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

#### BIKE DATASET #####
input_path = '/Users/alemshaimardanov/PycharmProjects/Stocks_forecasting/base_ml_and_stats_models'
input_files_list = ['2013-06-citibike-tripdata.csv', '2013-07-citibike-tripdata.csv', '2013-08-citibike-tripdata.csv']
# input_files_list = ['2013-06-citibike-tripdata.csv']
output_path = '/Users/alemshaimardanov/PycharmProjects/Stocks_forecasting/base_ml_and_stats_models/bikes'
bike_stations = ['W 20 St & 11 Ave','E 17 St & Broadway']
### all_months_df = create_bike_df(input_path, input_files_list, output_path, bike_stations)
all_months_df = pd.read_csv(output_path + '/' + 'Bikes.csv').dropna()

all_months_sum1= all_months_df['Value1'].sum()
all_months_sum2= all_months_df['Value2'].sum()
print('All months Shape: ', all_months_df.shape)
print('All months sum Bikes Count 1: ', all_months_sum1)
print('All months sum Bikes Count 2: ', all_months_sum2)
timeseries_name = 'CitiBikes'
output_plots_path = output_path + '/plots'


########################

# ##### S&P 100 STOCKS DATASET #####################################################
# input_path = '/Users/alemshaimardanov/PycharmProjects/Stocks_forecasting/base_ml_and_stats_models/s_and_p_stocks'
# input_file = 'NVDA.csv' # Indicate a file with a certain stock's price data 
# timeseries_name = 'NVDA'
# output_path = '/Users/alemshaimardanov/PycharmProjects/Stocks_forecasting/base_ml_and_stats_models/s_and_p_stocks'
# output_plots_path = output_path + '/plots'
# all_months_df = create_stock_df(input_path, input_file, output_path).dropna() # Create a df from input files and drop NaN values
################################################################################################


# is multivariate?
is_multivariate_input = True
columns = list(all_months_df.columns)
if len(columns) > 2:
    is_multivariate_input = True

#########################

# Flag to determine whether to call ensemble method or not
call_ensemble = False
# base_models_list = ['ar', 'arima','theta','rfr','lstm','var'] # Default list of base models
base_models_list = ['ar', 'arima','theta','var']

# Split into train and test data
if is_multivariate_input:
    train, test = train_test_split(all_months_df, test_size=0.2, shuffle=False) # IF MULTIVARIATE TS
    print('Type of train: ', type((train)))
    print('Len of train: ', train.shape[0])
    print('Len of test: ', test.shape[0])

    ###
    # predictions, y_test, score = var_bike_predict(train, test, output_plots_path, timeseries_name, 3)
    # print('VAR r2-score: ', score)

    ###

else:
    train, test = train_test_split(all_months_df['Value1'], test_size=0.2, shuffle=False) # IF UNIVARIATE TS
    print('Type of train: ', type((train)))
    print('Len of train: ', train.shape[0])
    print('Len of test: ', test.shape[0])
    # predictions, y_test, score = var_predict(train, test, output_plots_path, timeseries_name, 3)
    # print('VAR r2-score: ', score)
# print('name of columns: ', list(train.columns))

# Make predictions
# predictions, y_test, score = var_predict(train, test, output_plots_path, timeseries_name, 3)
# print('VAR r2-score: ', score)



# Call rfr_ensemble
base_models_df = pd.read_csv(output_path + '/' + 'base_model_predictions.csv')
base_models_df.drop(['Unnamed: 0'], axis = 1, inplace = True) 
base_model_predictions = base_models_df.to_dict('dict')

ensemble_predictions, y_test, ensemble_score = rfr_ensemble_bike_predict(train, test, output_plots_path, timeseries_name, base_model_predictions)
print('RF ensemble r2-score: ', ensemble_score)

# Ensemble method
if call_ensemble:
    # base_model_predictions = [] # IF FINANCIAL time series
    base_model_predictions = {}
    for base_model in base_models_list:

        if base_model == 'ar':
            predictions, y_test, score = ar_bike_predict(train, test, output_plots_path, timeseries_name)
            print('AR1 r2-score: ', score)
            base_model_predictions['ar'] = predictions
        elif base_model == 'arima':
            predictions, y_test, score = arima_bike_predict(train, test, output_plots_path, timeseries_name)
            print('ARIMA r2-score: ', score)
            base_model_predictions['arima'] = predictions
        elif base_model == 'theta':
            predictions, y_test, score = theta_bike_predict(train, test, output_plots_path, timeseries_name)
            print('Theta r2-score: ', score)
            base_model_predictions['theta'] = predictions
        elif base_model == 'var':
            predictions, y_test, score = var_bike_predict(train, test, output_plots_path, timeseries_name, 3)
            print('VAR r2-score: ', score)
            base_model_predictions['var'] = predictions
        elif base_model == 'rfr':
            predictions, y_test, score = rfr_predict(train, test, output_plots_path, timeseries_name)
            print('RFR r2-score: ', score)
            base_model_predictions['rfr'] = predictions
        elif base_model == 'lstm':
            predictions, y_test, score = lstm_predict(train, test, output_plots_path, timeseries_name)
            print('LSTM r2-score: ', score)
            base_model_predictions['lstm'] = predictions
        # base_model_predictions.append(predictions) # IF FINANCIA time series


    # print('Size of bm_preds: ',base_model_predictions)
    print('Size of [0]', len(base_model_predictions['ar']))
    print('Size of [2]', len(base_model_predictions['var']))
    # Call rfr_ensemble
    ensemble_predictions, y_test, ensemble_score = rfr_ensemble_bike_predict(train, test, output_plots_path, timeseries_name, base_model_predictions)
    print('RF ensemble r2-score: ', ensemble_score)
print('End of program')