import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# align time series data and knowledge graph data
def align_ts_kg(train_ts_path, test_ts_path, kg_path):

    # load ts data
    train_ts_data = pickle.load(open(train_ts_path, 'rb'))
    test_ts_data = pickle.load(open(test_ts_path, 'rb'))

    # load kg data
    KG_dict = pickle.load(open(kg_path, 'rb'))


    # align train data, according to the 'lat' and 'lon'
    train_data = {}
    test_data = {}

    for sensor in train_ts_data.keys():
        ts_lat = train_ts_data[sensor]['lat']
        ts_lon = train_ts_data[sensor]['lon']

        for kg_sensor in KG_dict.keys():
            kg_lat = KG_dict[kg_sensor]['lat']
            kg_lon = KG_dict[kg_sensor]['lon']

            sensor_dict = {}

            if ts_lat == kg_lat and ts_lon == kg_lon:
                sensor_dict['data'] = train_ts_data[sensor]['data']
                sensor_dict['amenities_count_list'] = KG_dict[kg_sensor]['amenities_count_list']
                sensor_dict['graph_matrix_1'] = KG_dict[kg_sensor]['graph_matrix_1']
                sensor_dict['graph_matrix_2'] = KG_dict[kg_sensor]['graph_matrix_2']

                train_data[sensor] = sensor_dict

            else:
                continue

    # align test data, according to the 'lat' and 'lon'
    for sensor in test_ts_data.keys():
        ts_lat = test_ts_data[sensor]['lat']
        ts_lon = test_ts_data[sensor]['lon']

        for kg_sensor in KG_dict.keys():
            kg_lat = KG_dict[kg_sensor]['lat']
            kg_lon = KG_dict[kg_sensor]['lon']

            sensor_dict = {}

            if ts_lat == kg_lat and ts_lon == kg_lon:
                sensor_dict['data'] = test_ts_data[sensor]['data']
                sensor_dict['amenities_count_list'] = KG_dict[kg_sensor]['amenities_count_list']
                sensor_dict['graph_matrix_1'] = KG_dict[kg_sensor]['graph_matrix_1']
                sensor_dict['graph_matrix_2'] = KG_dict[kg_sensor]['graph_matrix_2']

                test_data[sensor] = sensor_dict

            else:
                continue

    return train_data, test_data


# create dataset
def create_dataset(train_data, test_data, window, step):
    # create train dataset
    train_X = []
    train_Y = []

    for sensor in train_data.keys():

        train_ts_x_y = np.vstack((train_data[sensor]['data']['trend'].to_numpy(), train_data[sensor]['data']['seasonal'].to_numpy(), train_data[sensor]['data']['resid'].to_numpy())).T
        # use slide window to create train data
        train_ts_x = []
        train_ts_y = []

        ts_len = train_data[sensor]['data'].shape[0]

        for i in range(ts_len - window):
            if i % step == 0:
                train_ts_x.append(train_ts_x_y[i:i+window, :])
                train_ts_y.append(train_ts_x_y[i+window:i+window+step, :])

        train_kg_x = torch.from_numpy(np.array(train_data[sensor]['amenities_count_list']).reshape(28, 1)).float()
        train_kg_1 = torch.from_numpy(train_data[sensor]['graph_matrix_1']).float()
        train_kg_2 = torch.from_numpy(train_data[sensor]['graph_matrix_2']).float()

        for i in range(len(train_ts_x)):
            train_x = [torch.from_numpy(train_ts_x[i]).float(), train_kg_x, train_kg_1, train_kg_2]
            train_y = torch.from_numpy(train_ts_y[i]).float()

            train_X.append(train_x)
            train_Y.append(train_y)


    # create test dataset
    test_X = []
    test_Y = []

    for sensor in test_data.keys():

        test_ts_x_y = np.vstack((test_data[sensor]['data']['trend'].to_numpy(), test_data[sensor]['data']['seasonal'].to_numpy(), test_data[sensor]['data']['resid'].to_numpy())).T
        # use slide window to create test data
        test_ts_x = []
        test_ts_y = []
        ts_len = test_data[sensor]['data'].shape[0]

        for i in range(ts_len - window):
            if i % step == 0:
                test_ts_x.append(test_ts_x_y[i:i+window, :])
                test_ts_y.append(test_ts_x_y[i+window:i+window+step, :])

        test_kg_x = torch.from_numpy(np.array(test_data[sensor]['amenities_count_list']).reshape(28, 1)).float()
        test_kg_1 = torch.from_numpy(test_data[sensor]['graph_matrix_1']).float()
        test_kg_2 = torch.from_numpy(test_data[sensor]['graph_matrix_2']).float()

        for i in range(len(test_ts_x)):
            test_x = [torch.from_numpy(test_ts_x[i]).float(), test_kg_x, test_kg_1, test_kg_2]
            test_y = torch.from_numpy(test_ts_y[i]).float()

            test_X.append(test_x)
            test_Y.append(test_y)

    return train_X, train_Y, test_X, test_Y


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x_ts = self.x_data[index][0]
        x_f = self.x_data[index][1]
        kg_1 = self.x_data[index][2]
        kg_2 = self.x_data[index][3]
        y = self.y_data[index]
        return x_ts, x_f, kg_1, kg_2, y
    

def metric_test(pred, real):

    # sum 3 items to get flow
    pred_flow = pred.reshape(-1, 3)[:, 0] + pred.reshape(-1, 3)[:, 1] + pred.reshape(-1, 3)[:, 2]
    real_flow = real.reshape(-1, 3)[:, 0] + real.reshape(-1, 3)[:, 1] + real.reshape(-1, 3)[:, 2]

    # mse
    mse = mean_squared_error(real_flow, pred_flow)
    # rmse
    rmse = np.sqrt(mse)
    # mae
    mae = mean_absolute_error(real_flow, pred_flow)
    # r2
    r2 = r2_score(real_flow, pred_flow)

    return mse, rmse, mae, r2