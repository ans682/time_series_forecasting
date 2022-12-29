import os
import pandas as pd
from collections import defaultdict
import numpy as np
current_path = os.getcwd()
print("Current path: ",current_path)

from ..helpers.helpers import count_hourly_demand


# Global function to process citibike raw dataset
def create_bike_df(file_path, file_names_list, output_path, bike_stations, input_file_name, is_bike_ts_valid=False):
    if is_bike_ts_valid:
        all_months_df = pd.read_csv(output_path + '/' + input_file_name).dropna()
        return all_months_df

    else:
        final_df = pd.DataFrame()
        all_months_df = pd.DataFrame(columns=['Date and Hour', 'Bikes Count'])
        bike1_df = pd.DataFrame(columns=['Date and Hour', 'Bikes Count 1'])
        bike2_df = pd.DataFrame(columns=['Date and Hour', 'Bikes Count 2'])

        i = 1
        for bike_station in bike_stations:
            
            for file_name in file_names_list:
                # Load Citi bike csv into dataframe
                bike_df = pd.read_csv(file_path + '/' + file_name)
                
                # Count frequency of every station.
                # most_popular_station = bike_df['start station name'].value_counts().index.tolist()[0]

                # Extract rows about the most popular bike station into dataframe
                most_popular_station_df = bike_df.loc[bike_df['start station name'] == bike_station]

                # Create a copy of starttimes of the most popular station
                raw_demand = most_popular_station_df.loc[:,'starttime'].copy()

                # Sort time stamps in place
                raw_demand.sort_values(ascending=True, inplace=True)

                # Convert the sorted time stamps into dataframe
                raw_demand_df = pd.DataFrame(raw_demand)

                # Get the period of the dataframe and have it in the YYYY-MM format
                file_period = most_popular_station_df['starttime'].iloc[0].split()[0].split('-')
                file_period = '-'.join(file_period[:2])

                # Save the dataframe into CSV file
                # output_path = current_path + '/datasets/bikes'
                raw_demand_df.to_csv(output_path + '/' + file_period + '.csv', index=False)

                # Call the function to count demand of bikes by hour
                bike_hourly_demand = count_hourly_demand(raw_demand)

                total_count = 0
                hours_in_1_day = 0
                pick_ups_in_1_day = 0

                month_bikes_list = []
                
                for date, hours in bike_hourly_demand.items():
                    hours_in_1_day = 0
                    pick_ups_in_1_day = 0
                    
                    for hour, count in hours.items():
                        hours_in_1_day += 1
                        pick_ups_in_1_day += count
                        total_count += count
                        month_bikes_list.append([date + ' ' + hour, count])

                print('Total count: ', total_count)

                # Convert month_bike_list to df
                month_bikes_df = pd.DataFrame(month_bikes_list, columns=['Date and Hour', 'Bikes Count'])

                # Add month_bikes_df to final time series df
                all_months_df = pd.concat([all_months_df, month_bikes_df], ignore_index=True, axis=0)
            
            if i == 1:
                final_df['Date'] = all_months_df['Date and Hour']
                final_df['Value1'] = all_months_df['Bikes Count']
            else:
                final_df['Value2'] = all_months_df['Bikes Count']

            i += 1

            all_months_df = pd.DataFrame(columns=['Date and Hour', 'Bikes Count 1', 'Bikes Count 2'])
        
        final_df.to_csv(output_path + '/' + 'Bikes' + '.csv', index=False)
        return final_df




