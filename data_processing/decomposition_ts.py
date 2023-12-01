import pandas as pd
import pickle
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split


# load raw traffic data
dublin_complete = pd.read_csv('GAN-GNN/dataset/dublin_raw_data.csv')

# get the unique sensor ids
sensor_ids = dublin_complete['Site'].unique()


# creat a dict to save the data
sensor_data = {}

for sensor_id in sensor_ids:

        # creat a dict for each sensor, and save the lat and lon
        sensor_data[sensor_id] = {}
        sensor_data[sensor_id]['lat'] = dublin_complete[dublin_complete['Site']==sensor_id]['Lat'].unique()[0]
        sensor_data[sensor_id]['lon'] = dublin_complete[dublin_complete['Site']==sensor_id]['Long'].unique()[0]
        
        # get the data
        sensor_data[sensor_id]['data'] = dublin_complete[dublin_complete['Site']==sensor_id][['End_Time', 'Sum_Volume']]

        # drop duplicates
        sensor_data[sensor_id]['data'] = sensor_data[sensor_id]['data'].drop_duplicates(subset=['End_Time'])



start_time = pd.to_datetime('2022-04-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
end_time = pd.to_datetime('2022-06-30 23:00:00', format='%Y-%m-%d %H:%M:%S')


# reindex and padding with interpolation
for sensor_id in sensor_data.keys():

    # get the data
    df = sensor_data[sensor_id]['data']

    # set the index
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('End_Time')

    # generate new time index
    time_index = pd.date_range(start_time, end_time, freq='1H')

    # reindex
    df = df.reindex(time_index)

    # padding with interpolation
    df = df.interpolate(method='linear', limit_direction='both')
    
    # save the data
    sensor_data[sensor_id]['data'] = df


# if 'Sum_Volume' of sensor is all of 0, then drop it
new_sensor_data = {}

for sensor_id in sensor_data.keys():
    if sensor_data[sensor_id]['data']['Sum_Volume'].sum() != 0:
        new_sensor_data[sensor_id] = sensor_data[sensor_id]



# split sensors into train set and test set, 80% for train, 20% for test
train_ids, test_ids = train_test_split(list(new_sensor_data.keys()), test_size=0.2, random_state=42)

train_data = {k: new_sensor_data[k] for k in train_ids}
test_data = {k: new_sensor_data[k] for k in test_ids}


# for train data
for sensor_id in train_data.keys():
    flow = train_data[sensor_id]['data']['Sum_Volume']

    # normalize the flow
    flow_nor = (flow - flow.min()) / (flow.max() - flow.min()) * 2 - 1

    # using STL to decompose the time series
    res = STL(flow_nor, period=24*7, seasonal=15).fit()     # set period as one week, seasonal should be an odd number
    train_data[sensor_id]['data']['flow_nor'] = flow_nor
    train_data[sensor_id]['data']['trend'] = res.trend
    train_data[sensor_id]['data']['seasonal'] = res.seasonal
    train_data[sensor_id]['data']['resid'] = res.resid

# for test data
for sensor_id in test_data.keys():
    flow = test_data[sensor_id]['data']['Sum_Volume']

    flow_nor = (flow - flow.min()) / (flow.max() - flow.min()) * 2 - 1

    # using STL
    res = STL(flow_nor, period=24*7, seasonal=15).fit() # 1 week
    test_data[sensor_id]['data']['flow_nor'] = flow_nor
    test_data[sensor_id]['data']['trend'] = res.trend
    test_data[sensor_id]['data']['seasonal'] = res.seasonal
    test_data[sensor_id]['data']['resid'] = res.resid



# save the data
with open('GAN-GNN/data_processing/train_ts_dublin.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('GAN-GNN/data_processing/test_ts_dublin.pkl', 'wb') as f:
    pickle.dump(test_data, f)
