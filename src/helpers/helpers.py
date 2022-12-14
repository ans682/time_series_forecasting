from collections import defaultdict

# Define a function to determine next hour from timestamp
def get_next_hour(hour):
    next_hour = ''
    if hour[0] == '0':
        if hour[1] == '9':
            next_hour = '10'
        else:
            next_hour = '0' + str(int(hour[1]) + 1)
            
    elif hour[0] == '1':
        if hour[1] == '9':
            next_hour = '20'
        else:
            next_hour = '1' + str(int(hour[1]) + 1)
            
    else:
        if hour[1] == '3':
            next_hour = '00'
        else:
            next_hour = '2' + str(int(hour[1]) + 1)
    return next_hour


# Define a function to generate a 24 h dict such that key=hour, value=count of hours
def create_24_h_dict():
    one_day_dict = {
        '00':0,'01':0,'02':0,'03':0,'04':0,'05':0,'06':0,'07':0,'08':0,'09':0,
        '10':0,'11':0,'12':0,'13':0,'14':0,'15':0,'16':0,'17':0,'18':0,'19':0,
        '20':0,'21':0,'22':0,'23':0
    }
    return one_day_dict


# Define a function to extract hour from start time
def get_date_hour(starttime):
    hour = starttime.split()[1].split(':')[0]
    date = starttime.split()[0]
    return date, hour


# Define a function to group starttime timestamps by 1 hour
def count_hourly_demand(raw_demand):
    prev_hour = raw_demand.iloc[0].split()[1].split(':')[0]
    prev_date = raw_demand.iloc[0].split()[0]
    next_hour = get_next_hour(prev_hour)
    
    bike_hourly_demand = defaultdict(str)
    bike_hourly_demand = {prev_date: create_24_h_dict()}

    i = 0
    for index, value in raw_demand.items():
        cur_date, cur_hour = get_date_hour(value)
        # If you're processing the same date
        if prev_date == cur_date: 
            # If you're processing the same hour, update the counter
            if prev_hour == cur_hour:
                bike_hourly_demand[prev_date][prev_hour] += 1
            # Otherwise, this is a new hour. Add a new hour key to the dict
            else:
                bike_hourly_demand[prev_date][cur_hour] = 1
                prev_hour = cur_hour

        # Otherwise, this is a new date. 
        else:
            # This is a new hour. Hence, add a new date key to the dict
            bike_hourly_demand[cur_date] = create_24_h_dict()
            bike_hourly_demand[cur_date][cur_hour] += 1
            prev_hour = cur_hour
            prev_date = cur_date
            
    return bike_hourly_demand


# Create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff