import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, MSELoss
from torch_geometric.nn import SAGEConv, NNConv, GATConv, GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import time
import math
import winsound
import torch
import torch.nn as nn

st = time.process_time()


class Network(torch.nn.Module):

    def __init__(self, n_node_features, node_embedding_size, model):
        super(Network, self).__init__()
        self.conv1 = None
        self.conv2 = None

        if model == 'GCN':
            self.conv1 = GCNConv(n_node_features, node_embedding_size)
        if model == 'GAT':
            self.conv1 = GCNConv(n_node_features, node_embedding_size, heads=1)

    def forward(self, node_features, edge_index_, edge_weight_):
        x = self.conv1(node_features, edge_index_, edge_weight_)
        x = F.relu(x)
        x = F.dropout(x, p=0.001, training=self.training)
        return x


class MLP(torch.nn.Module):
    def __init__(self, n_node_features, hidden_channels):
        super(MLP, self).__init__()
        self.lin1 = Linear(2 * n_node_features + 1, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.lin1(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.001, training=self.training)
        # x = torch.sigmoid(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x


mseresult = []
r2result = []
maeresult = []
rmseresult = []
data_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
i = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97]
city = ['Anaheim']
for dataset in city:

    # rate = 10
    for num in data_num:
        for rate in i:

            data_attraction_address = 'E:/class/Ewh, University!/8/summer project/TDM/Data/' + str(
                dataset) + '/Scaled/' + 'prepared_data-' + str(
                num) + '/' + str(rate) + '/' + 'attraction.csv '
            data_production_address = 'E:/class/Ewh, University!/8/summer project/TDM/Data/' + str(
                dataset) + '/Scaled/' + 'prepared_data-' + str(
                num) + '/' + str(rate) + '/' + 'production.csv'
            data_travel_time_address = 'E:/class/Ewh, University!/8/summer project/TDM/Data/' + str(
                dataset) + '/Scaled/' + 'prepared_data-' + str(
                num) + '/' + str(rate) + '/' + 'travel_time_matrix.csv'

            data_od_train_address = 'E:/class/Ewh, University!/8/summer project/TDM/Data/' + str(
                dataset) + '/Scaled/' + 'prepared_data-' + str(
                num) + '/' + str(rate) + '/' + 'at_miss' + str(
                int(rate) / int(100)) + '0_train_od_matrix.csv'
            data_od_test_address = 'E:/class/Ewh, University!/8/summer project/TDM/Data/' + str(
                dataset) + '/Scaled/' + 'prepared_data-' + str(
                num) + '/' + str(rate) + '/' + 'at_miss' + str(
                int(rate) / int(100)) + '0_test_od_matrix.csv'
            data_od_valid_address = 'E:/class/Ewh, University!/8/summer project/TDM/Data/' + str(
                dataset) + '/Scaled/' + 'prepared_data-' + str(
                num) + '/' + str(rate) + '/' + 'at_miss' + str(
                int(rate) / int(100)) + '0_val_od_matrix.csv'

            # Import data
            data_attraction = pd.read_csv(data_attraction_address)
            data_production = pd.read_csv(data_production_address)
            data_travel_time = pd.read_csv(data_travel_time_address)

            data_od_train = pd.read_csv(data_od_train_address)
            data_od_test = pd.read_csv(data_od_test_address)
            data_od_valid = pd.read_csv(data_od_valid_address)

            # Fix the problem of removed rows
            all_nodes = range(1, data_attraction['Unnamed: 0'].max() + 1)
            df = pd.DataFrame({'Unnamed: 0': all_nodes})
            data_attraction = pd.merge(df, data_attraction, on='Unnamed: 0', how='left').fillna(0)
            all_nodes = range(1, data_production['Unnamed: 0'].max() + 1)
            df = pd.DataFrame({'Unnamed: 0': all_nodes})
            data_production = pd.merge(df, data_production, on='Unnamed: 0', how='left').fillna(0)

            # Convert df to numpy array
            data_attraction = data_attraction['0'].to_numpy()
            data_production = data_production['0'].to_numpy()

            # Convert Travel Time from matrix format to table
            data_travel_time.reset_index(inplace=True)
            data_travel_time.rename(columns={'Unnamed: 0': 'src'}, inplace=True)
            data_travel_time = pd.melt(data_travel_time, id_vars=['src'], var_name='dst', value_name='time')
            data_travel_time = data_travel_time[data_travel_time['dst'] != 'index']
            data_travel_time['dst'] = data_travel_time['dst'].astype('int')
            data_travel_time = data_travel_time[data_travel_time['src'] != data_travel_time['dst']]

            # Convert OD from matrix format to table
            data_od_train.reset_index(inplace=True)
            data_od_train.rename(columns={'Unnamed: 0': 'src'}, inplace=True)
            data_od_train = pd.melt(data_od_train, id_vars=['src'], var_name='dst', value_name='demand')
            data_od_train = data_od_train[data_od_train['dst'] != 'index']
            data_od_train['dst'] = data_od_train['dst'].astype('int')
            data_od_train = data_od_train[data_od_train['src'] != data_od_train['dst']]
            train_mask = (data_od_train['demand'] != 'False') & (data_od_train['demand'] != 'No_connection')
            train_mask = train_mask.to_numpy()
            data_od_train = data_od_train[train_mask]

            data_od_test.reset_index(inplace=True)
            data_od_test.rename(columns={'Unnamed: 0': 'src'}, inplace=True)
            data_od_test = pd.melt(data_od_test, id_vars=['src'], var_name='dst', value_name='demand')
            data_od_test = data_od_test[data_od_test['dst'] != 'index']
            data_od_test['dst'] = data_od_test['dst'].astype('int')
            data_od_test = data_od_test[data_od_test['src'] != data_od_test['dst']]
            test_mask = (data_od_test['demand'] != 'False') & (data_od_test['demand'] != 'No_connection')
            test_mask = test_mask.to_numpy()
            data_od_test = data_od_test[test_mask]

            data_od_valid.reset_index(inplace=True)
            data_od_valid.rename(columns={'Unnamed: 0': 'src'}, inplace=True)
            data_od_valid = pd.melt(data_od_valid, id_vars=['src'], var_name='dst', value_name='demand')
            data_od_valid = data_od_valid[data_od_valid['dst'] != 'index']
            data_od_valid['dst'] = data_od_valid['dst'].astype('int')
            data_od_valid = data_od_valid[data_od_valid['src'] != data_od_valid['dst']]
            valid_mask = (data_od_valid['demand'] != 'False') & (data_od_valid['demand'] != 'No_connection')
            valid_mask = valid_mask.to_numpy()
            data_od_valid = data_od_valid[valid_mask]

            # Add fake dummy row at the top
            data_attraction = np.append(0, data_attraction)
            data_production = np.append(0, data_production)

            # expand dim
            data_attraction = np.expand_dims(data_attraction, axis=1)
            data_production = np.expand_dims(data_production, axis=1)

            # node feature
            node_feature = np.concatenate((data_production, data_attraction), axis=1)
            X = torch.from_numpy(node_feature)
            X = X.to(torch.float)

            # src dst nodes
            src_nodes_train = data_od_train['src'].to_numpy().tolist()
            dst_nodes_train = data_od_train['dst'].to_numpy().tolist()
            src_nodes_valid = data_od_valid['src'].to_numpy().tolist()
            dst_nodes_valid = data_od_valid['dst'].to_numpy().tolist()
            src_nodes_test = data_od_test['src'].to_numpy().tolist()
            dst_nodes_test = data_od_test['dst'].to_numpy().tolist()

            # edge index
            edge_index_train = torch.tensor([src_nodes_train, dst_nodes_train], dtype=torch.long)
            edge_index_valid = torch.tensor([src_nodes_valid, dst_nodes_valid], dtype=torch.long)
            edge_index_test = torch.tensor([src_nodes_test, dst_nodes_test], dtype=torch.long)

            # edge weight (travel time)
            times = data_travel_time['time'].to_numpy().tolist()
            edge_weight = 0.001 / torch.tensor([times], dtype=torch.float)
            edge_weight = torch.squeeze(edge_weight)
            scaler_edge_weight = StandardScaler()
            edge_weight_scaled = scaler_edge_weight.fit_transform(edge_weight.reshape(-1, 1))
            edge_weight_scaled = torch.tensor(edge_weight_scaled).to(torch.float)

            # y
            y_train = data_od_train['demand'].to_numpy().astype(float)
            y_test = data_od_test['demand'].to_numpy().astype(float)
            y_valid = data_od_valid['demand'].to_numpy().astype(float)

            # Input data to GNN
            data = Data(x=X, edge_index=edge_index_train)

            # Model/Optimizer
            device = torch.device('cpu')
            data = data.to(device)
            model = Network(n_node_features=data.num_node_features, node_embedding_size=8, model='GAT')
            model_name = 'GAT'
            model = model.to(device)
            link_regressor = MLP(n_node_features=8, hidden_channels=64)
            link_regressor = link_regressor.to(device)
            params = list(model.parameters()) + list(link_regressor.parameters())
            optimizer = torch.optim.Adam(params, lr=0.015, weight_decay=0.0000001)
            loss = MSELoss()

            y_train = torch.from_numpy(y_train).to(torch.float)

            # edge index split
            edge_index_train = torch.tensor([edge_index_train[0].tolist(),
                                             edge_index_train[1].tolist()])

            edge_index_train = edge_index_train.to(torch.long).to(device)
            src_train, dst_train = edge_index_train

            edge_index_valid = torch.tensor([edge_index_valid[0].tolist(),
                                             edge_index_valid[1].tolist()])

            edge_index_valid = edge_index_valid.to(torch.long).to(device)
            src_valid, dst_valid = edge_index_valid

            edge_index_test = torch.tensor([edge_index_test[0].tolist(),
                                            edge_index_test[1].tolist()])

            edge_index_test = edge_index_test.to(torch.long).to(device)
            src_test, dst_test = edge_index_test

            # edge weight split
            edge_weight_train = torch.tensor([edge_weight[train_mask].tolist()])
            edge_weight_train = torch.unsqueeze(edge_weight_train[0], dim=1)
            edge_weight_train = edge_weight_train.to(torch.float).to(device)

            edge_weight_valid = torch.tensor([edge_weight[valid_mask].tolist()])
            edge_weight_valid = torch.unsqueeze(edge_weight_valid[0], dim=1)
            edge_weight_valid = edge_weight_valid.to(torch.float).to(device)

            edge_weight_test = torch.tensor([edge_weight[test_mask].tolist()])
            edge_weight_test = torch.unsqueeze(edge_weight_test[0], dim=1)
            edge_weight_test = edge_weight_test.to(torch.float).to(device)

            min_loss_val = np.Inf
            best_epoch = 0

            loss_train = list()
            loss_valid = list()

            best_model = None
            best_regressor = None

            best_loss = float('inf')
            patient = 20
            # Train Loop
            for epoch in range(1, 801):
                model.train()
                link_regressor.train()
                optimizer.zero_grad()

                output = model(node_features=X, edge_index_=edge_index_train, edge_weight_=edge_weight_train)
                output_2 = output.detach().numpy()
                regressor_input = torch.cat((output[src_train], output[dst_train], edge_weight_train), 1)

                y_pred_train = link_regressor(regressor_input)
                y_pred_train = torch.squeeze(y_pred_train)
                y_true_train = torch.squeeze(y_train)

                loss_ = loss(y_pred_train, y_true_train)
                loss_.backward()
                optimizer.step()
                model.eval()
                link_regressor.eval()
                regressor_input = torch.cat((output[src_valid], output[dst_valid], edge_weight_valid), 1)
                y_pred_valid = link_regressor(regressor_input)
                y_pred_valid = torch.squeeze(y_pred_valid).detach().numpy()
                loss_val = mean_squared_error(y_valid, y_pred_valid)
                if loss_val < min_loss_val:
                    min_loss_val = loss_val
                    best_epoch = epoch
                    best_model = model.state_dict()
                    best_regressor = link_regressor.state_dict()

                if loss_val <= best_loss:
                    best_loss = loss_val
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= patient:
                    print(f"Early_stop_at {epoch} Epoch")
                    break

                loss_train.append(loss_.item())
                loss_valid.append(loss_val)

            print(epoch, 'Loss train: ', str(loss_.item()), 'Loss valid: ', str(loss_val))

            print('Loss valid: ', str(loss_val))
            print('GNN- Best Epoch:', str(best_epoch))

            model.load_state_dict(best_model)
            link_regressor.load_state_dict(best_regressor)
            model.eval()
            link_regressor.eval()
            with torch.inference_mode():
                # Test the best model
                output = model(node_features=X, edge_index_=edge_index_train, edge_weight_=edge_weight_train)
                regressor_input = torch.cat((output[src_test], output[dst_test], edge_weight_test), 1)
                y_pred = link_regressor(regressor_input)
                y_pred = torch.squeeze(y_pred)
                y_pred = y_pred.detach().numpy()
                rmse = mean_squared_error(y_test, y_pred, squared=True)
                mse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                # print(y_test)
                rmseresult.append(rmse)
                maeresult.append(mae)
                r2result.append(r2)
                mseresult.append(mse)
            print('RMSE Test: ', str(rmse))
            print('MSE Test: ', str(mse))
            print('MAE Test: ', str(mae))
            print('R2 Test: ', str(r2))
            print(rate)

        print('rmseresult = ', rmseresult)
        print('mseresult = ', mseresult)
        print('r2result = ', r2result)
        print('maeresult = ', maeresult)
        print(len(rmseresult))

        et = time.process_time()
        res = et - st
        print(res, 'sec')

winsound.Beep(1500, 2000)
