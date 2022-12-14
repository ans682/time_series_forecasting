import os
import pandas as pd
import numpy as np

def create_stock_df(file_path, input_file, output_path):
    timeseries_names = []

    
    # # Get the name of the file and add it to the list of the timeseries names
    # ts_name = file_name.split()[0]
    # timeseries_names.append(ts_name)

    # Load the csv file into training_data df
    training_data = pd.read_csv(file_path + '/' + input_file)
    columns = list(training_data.columns)

    if len(columns) <= 2:
        training_data.columns = ['Date', 'Value1']
        df = pd.DataFrame({"Date": training_data.index, "Value1": training_data['Value1']})
        return df
    else:
        training_data.columns = ['Date', 'Value1', 'Value2']
        df = pd.DataFrame({"Date": training_data.index, "Value1": training_data['Value1'], "Value2": training_data['Value2']})
        return df
    # print('Column names: ', columns)
    
    # all_months_df = pd.DataFrame(columns=['Date', 'Value']) # Use this for Univariate ts model
    # all_months_df = pd.DataFrame(columns=['Date', 'Value1', 'Value2']) # Use this for Multivariate ts model
    # # all_months_df = pd.concat([all_months_df, training_data], ignore_index=True, axis=0)
    

    # # Create a dataframe with the stock prices
    # df = pd.DataFrame({"Date": training_data.index, "Value1": training_data['Value1'], "Value2": training_data['Value2']})
    # # print(df)
    # print('Exiting load stock file')
    # return df

