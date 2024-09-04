# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:09:12 2023

@author: yrt05
"""


import scipy.sparse
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch_geometric
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import HeteroData
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import HeteroConv, GraphConv, Linear, GCNConv
from torch.nn import Parameter
import pickle
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import degree
from torch_geometric.transforms import NormalizeFeatures, Compose


torch.autograd.set_detect_anomaly(True)



with open("Data/data_06_20_index.pickle", 'rb') as f:
    data_temp = pickle.load(f)

import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

seed = 333
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
# import numpy as np
np.random.seed(seed)

#%% Remove nodes
data_temp

# covid_indices_in_A_bar_100_nodes
indices_in_A_bar = torch.load("Data/Zika_indices_in_A_bar_100_nodes.pth")
torch.unique(indices_in_A_bar)

indices_in_A_bar = indices_in_A_bar.long()

data = data_temp.clone()
# Remove nodes and associated edges

node_mask_prot = torch.ones(data['Prot'].x.size(0), dtype=torch.bool)
node_mask_drug = torch.ones(data['Drug'].x.size(0), dtype=torch.bool)

for node_type in data.node_types:

        
        if node_type == 'Prot':
            for node_id in indices_in_A_bar: # nodes to remove
                node_mask_prot[torch.arange(data[node_type].x.size(0)) == node_id] = False
                # print(torch.all(node_mask_prot))
    #%% remove edge
    
        for edge_type, edge_index in data.edge_index_dict.items():
            # Filter out edges connected to removed nodes
            edge_current = edge_index.clone()
            
            if edge_type == ('Prot', 'interact', 'Prot') or edge_type == ('Prot', 'rev_interact', 'Prot'):
                # print(1)
                src_mask = node_mask_prot[edge_current[0]]
                # print('src_mask ' ,torch.all(src_mask))
                dst_mask = node_mask_prot[edge_current[1]]

            elif edge_type == ('Drug', 'interact', 'Drug') or edge_type == ('Drug', 'rev_interact', 'Drug'):
                
                src_mask = node_mask_drug[edge_current[0]]
                dst_mask = node_mask_drug[edge_current[1]]

            elif edge_type == ('Drug', 'bind', 'Prot'):
                
                src_mask = node_mask_drug[edge_current[0]]
                dst_mask = node_mask_prot[edge_current[1]]
                print('dst_mask ' ,torch.all(dst_mask))

            elif edge_type == ('Prot', 'rev_bind', 'Drug'):
                
                src_mask = node_mask_prot[edge_current[0]]
                print('src_mask ' ,torch.all(src_mask))
                dst_mask = node_mask_drug[edge_current[1]]
                         
            # Remove edges
            edge_mask = src_mask & dst_mask
            
            print('edge type ', edge_type)
            print('All True ', torch.all(edge_mask))

            
            # print('edge_mask.shape ', edge_mask.shape)
            # print('data[edge_type].shape ', (data[edge_type].edge_index).shape)
            print('edge num before: ', len(data[edge_type].edge_index[0]))
            
            data[edge_type].edge_index = edge_current[:, edge_mask].clone()
            
            print('edge num after: ', len(data[edge_type].edge_index[0]))

#%%
data_temp = data 
# %% To_undirect --> Split

data_temp = ToUndirected(merge=False)(data_temp)  # same edges
# Remove "reverse" label.
del data_temp['Prot', 'rev_interact', 'Prot'].edge_label
# Remove "reverse" label.
del data_temp['Drug', 'rev_interact', 'Drug'].edge_label
del data_temp['Prot', 'rev_bind', 'Drug'].edge_label  # Remove "reverse" label.

data_temp['Prot'].x

node_types = ['Prot', 'Drug']
transform_dict = {node_type: NormalizeFeatures() for node_type in node_types}
# transform = Compose(transform_dict)



#%% Apply normalization to each node type
for node_type in node_types:
    node_features = data_temp[node_type].x
    normalized_features = (  node_features - node_features.mean(dim=0)) / node_features.std(dim=0)
    data_temp[node_type].x = normalized_features

#%% Split data
train_data, val_data, test_data = RandomLinkSplit(is_undirected=False,
                                                  edge_types=data_temp.edge_types,
                                                  )(data_temp)


# %% Split --> To_undirect : Edge reverse

metadata = train_data.metadata()

x_prot = train_data['Prot']
x_drug = train_data['Drug']

group_name_concat = []
for item in metadata[1]:
    string = '__'.join(item)
    group_name_concat.append(string)


x_dict = train_data.x_dict

x_dict['Prot'].size()
x_dict['Drug'].size()


edge_index_dict = train_data.edge_index_dict
edge_label_index = train_data['Drug', 'Prot'].edge_label_index
# train_data['Prot', 'Drug'].edge_label_index


# global adj_mat_sub
data = train_data
edge_index_dict = data.edge_index_dict


# %% Adjacency all

def find_Adj(edge_index_dict):

    # edge_index_dict = data.edge_index_dict

    # PPI
    edge_PPI = edge_index_dict['Prot', 'interact', 'Prot']
    A_PPI = to_dense_adj(edge_PPI, max_num_nodes=x_prot['num_nodes'])[
        0].clamp(max=1)
    edge_PPI_t = edge_index_dict['Prot', 'rev_interact', 'Prot']
    A_PPI_t = to_dense_adj(edge_PPI_t, max_num_nodes=x_prot['num_nodes'])[
        0].clamp(max=1)
    A_PPI = (A_PPI + A_PPI_t).clamp(max=1)

    # DDI
    edge_DDI = edge_index_dict['Drug', 'interact', 'Drug']
    A_DDI = to_dense_adj(edge_DDI, max_num_nodes=x_drug['num_nodes'])[
        0].clamp(max=1)
    edge_DDI_t = edge_index_dict['Drug', 'rev_interact', 'Drug']
    A_DDI_t = to_dense_adj(edge_DDI_t, max_num_nodes=x_drug['num_nodes'])[
        0].clamp(max=1)
    A_DDI = (A_DDI + A_DDI_t).clamp(max=1)

    # Self loop
    A_PPI = (A_PPI + torch.eye(x_prot['num_nodes'])).clamp(max=1)
    A_DDI = (A_DDI + torch.eye(x_drug['num_nodes'])).clamp(max=1)

    # DTI
    edge_pair_DTI = edge_index_dict['Drug',
                                    'bind', 'Prot']  # DTI : drug [0], prot [1]
    edge_pair_DTI[0] = edge_pair_DTI[0] + x_prot['num_nodes']
    adj_mat_sub_DTI = to_dense_adj(
        edge_pair_DTI, max_num_nodes=x_prot['num_nodes'] + x_drug['num_nodes'])[0].clamp(max=1)
    edge_pair_DTI[0] = edge_pair_DTI[0] - x_prot['num_nodes']

    # IDI
    # DTI_t :  prot [0], drug [1]
    edge_pair_DTI_t = edge_index_dict['Prot', 'rev_bind', 'Drug']
    edge_pair_DTI_t[1] = edge_pair_DTI_t[1] + x_prot['num_nodes']
    adj_mat_sub_DTI_t = to_dense_adj(
        edge_pair_DTI_t, max_num_nodes=x_prot['num_nodes'] + x_drug['num_nodes'])[0].clamp(max=1)
    edge_pair_DTI_t[1] = edge_pair_DTI_t[1] - x_prot['num_nodes']

    Adj_all = torch.zeros(adj_mat_sub_DTI_t.shape)

    Adj_all[0:(A_PPI.shape[0]), 0:(A_PPI.shape[1])] = A_PPI
    Adj_all[(A_PPI.shape[0]):, (A_PPI.shape[1]):] = A_DDI

    Adj_all[(A_PPI.shape[1]):, 0:(A_PPI.shape[0])
            ] = adj_mat_sub_DTI[(A_PPI.shape[1]):, 0:(A_PPI.shape[0])]
    Adj_all[0:(A_PPI.shape[0]), (A_PPI.shape[1])
               :] = adj_mat_sub_DTI_t[0:(A_PPI.shape[0]), (A_PPI.shape[1]):]

    return Adj_all


# %% Define hyper parameter for verification use

def graph_property_hete(mydata):
    node_mean_0 = torch.mean(mydata['Prot'], dim=0).unsqueeze(0)
    node_mean_1 = torch.mean(mydata['Drug'], dim=0).unsqueeze(0)

    graph_property = torch.cat((node_mean_0, node_mean_1), dim=0)

    return graph_property

# %% My net

class Identity(nn.Module):
    def forward(self, x):
        return x


class GraphConvolution(nn.Module):
    # def __init__(self, num_node, in_features, hidden_channels, out_features, num_layers_before_merging, num_layers_after_merging):
    def __init__(self, hidden_channels, out_features, num_layer):

        super(GraphConvolution, self).__init__()

        random_seed = 2  ### same
        
        # random_seed = 77  ### Grid
        
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # num_node = 963

        in_features = hidden_channels
        out_features = hidden_channels

        num_layers_before_merging = 1
        num_layers_after_merging = num_layer - num_layers_before_merging

        self.leaky_relu = nn.LeakyReLU()
        # %% Before merging
        self.encoder_before_merging = torch.nn.ModuleList()
        for _ in range(num_layers_before_merging):
            conv = HeteroConv({
                edge_type: GraphConv(-1, hidden_channels,
                                     cached=True, bias = False)  # GraphConv
                # edge_type: GCNConv(-1, hidden_channels, cached = True, add_self_loops = False)  ### GCNConv
                for edge_type in metadata[1]
            })
            self.encoder_before_merging.append(conv)  # Add function here


        # %% After merging

        # self.hidden_layers_after_merge = nn.ModuleList([GCN(hidden_channels, hidden_channels) for _ in range(num_layers_after_merging)])
        self.num_layers_after_merging = num_layers_after_merging
        self.weight_matrices = nn.ParameterList()
        for i in range(num_layers_before_merging + num_layers_after_merging):
            if i == 0:
                self.weight_matrices.append(nn.Parameter(
                    torch.Tensor(in_features, hidden_channels)))
            elif i != num_layers_before_merging + num_layers_after_merging-1:
                self.weight_matrices.append(nn.Parameter(torch.Tensor(
                    hidden_channels, hidden_channels)))  # middle layer
            else:
                self.weight_matrices.append(nn.Parameter(
                    torch.Tensor(hidden_channels, out_features)))  # last layer



        self.layer_norm_prot = Identity()
        self.layer_norm_drug = Identity()
        
        self.layer_norm_identity = Identity()
        self.reset_parameters()
        self.intermediate_representations = []

    def reset_parameters(self):
        for weight in self.weight_matrices:
            
            
            nn.init.xavier_uniform_(weight)

# %% Forward path

    def forward(self, z_dict, edge_index_dict, edge_label_index):
        # def forward(self, z_dict, edge_index_dict):
        # %% First layer (before merging)
        z_dict_normalized = z_dict
        # z_dict_normalized = {key: self.layer_norm(x) for key, x in z_dict_normalized.items()}
        z_dict_normalized['Drug'] = self.layer_norm_drug(z_dict_normalized['Drug'])
        z_dict_normalized['Prot'] = self.layer_norm_prot(z_dict_normalized['Prot'])
        
        #%% Input normalization
        
        self.intermediate_representations = []  # latent features

        # %% Merging before (for same dimension)
        for conv in self.encoder_before_merging:  # encoder_before_merging = 1 for same dimension
            z_dict_normalized = conv(z_dict_normalized, edge_index_dict)

        # After merging
        # Feature: stack (drug + prot)
        z = torch.cat([z_dict_normalized['Prot'],
                      z_dict_normalized['Drug']], dim=0)

        # record first (merging) layer
        self.intermediate_representations.append(z)  # H_0

        # Large Adj matrix (with projection)

        # Graph feature
        graph_x_prot = z[0: (z_dict_normalized['Prot'].shape[0]), :]
        graph_x_drug = z[(z_dict_normalized['Prot'].shape[0])                         :, :]  # graph prot features

        # No graph features
        graph_x_prot = torch.zeros(graph_x_prot.shape)
        graph_x_drug = torch.zeros(graph_x_drug.shape)

        graph_property = torch.cat((graph_x_prot, graph_x_drug), dim=0)

        # %% A * X * W
        Adj_all = find_Adj(edge_index_dict)
        # Adj_all = Adj_all + torch.eye(Adj_all.shape[0], Adj_all.shape[1])

        # A normalize (mean aggregate)
        degrees_A_inv = torch.diag(torch.sum(Adj_all, dim=1)**(-1))
        Adj_all = degrees_A_inv @ Adj_all

        ### A * X
        z = Adj_all @ z  # torch.Size([963, 10]) project to same dimension

        all_x = torch.cat([z, graph_property], dim=0)
        # %% First layer (to put all_x together)
        all_x = self.layer_norm_identity(all_x @ self.weight_matrices[0])
        self.intermediate_representations.append(
            self.weight_matrices[0])  # W_0


        z_update = all_x[0: (z_dict_normalized['Prot'].shape[0] +
                             z_dict_normalized['Drug'].shape[0]), :]  # node features
        graph_x_update = all_x[(z_dict_normalized['Prot'].shape[0] +
                                z_dict_normalized['Drug'].shape[0]):, :]  # Graph feature

        self.intermediate_representations.append(z_update)  # H_1

        # %% Following layers

        for i in range(0, self.num_layers_after_merging):


            
            z_node_all = Adj_all @ z_update
            # node + graph 

            all_x = torch.cat([z_node_all, graph_x_update], dim=0)
            # all_x = (all_x @ weight_matrices[i+1]).relu()
            
            
            if i == self.num_layers_after_merging-1:
            # if i >= self.num_layers_after_merging-2:
                
                all_x = (all_x @ self.weight_matrices[i+1])  ### +1 due to 1 layer projection
                
            else:
                all_x = all_x @ self.weight_matrices[i+1]
                all_x = (all_x).relu()

              
            # node features
            z_update = all_x[0: (z_dict['Prot'].shape[0] +
                                 z_dict['Drug'].shape[0]), :]
            # Graph feature
            graph_x_update = all_x[( z_dict['Prot'].shape[0] + z_dict['Drug'].shape[0] ):, :]
                
            # print('graph_x_update', graph_x_update)
            # record updated latent features
            # W_1,W_2,...
            self.intermediate_representations.append(self.weight_matrices[i+1])
            self.intermediate_representations.append(z_update)  # H_2,H_2,...


        return z_update


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

    def forward(self, z, edge_label_index):
        z_update = z  # Prot + Drug   z_update.shape  z = z

        output = torch.sigmoid(torch.matmul(z_update, z_update.t()))

        A_adj_predicted = output
        
        return A_adj_predicted


# %% Model integration

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.encoder = GraphConvolution(
            hidden_channels, hidden_channels, num_layers)
        # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):  # "edge_label_index" not used
        z_dict = self.encoder(x_dict, edge_index_dict, edge_label_index)
        return self.decoder(z_dict, edge_label_index),   self.encoder.intermediate_representations
        # return self.decoder(z_dict, edge_label_index)

    # decompose "forward" into two parts

    def encode(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_label_index)
        return z_dict

    def decode(self, z, edge_label_index_combine, edge_label_combine):  # "edge_label_combine" not used
        # same as inner product
        specific_edge = ((z[edge_label_index_combine[0]] *
                         z[edge_label_index_combine[1]]).sum(dim=-1))
        specific_edge = torch.sigmoid(specific_edge)
        return specific_edge

# %% Train model
'''Use train and test and following functions in untitled1_hetero_all.py file '''

def negative_sampling_function(edge_index_dict, num_neg_samples, seed):
    # Identify zero entries in the adjacency matrix (potential negative samples)
    # test: edge_index_dict = train_data.edge_index_dict
    adj_matrix = find_Adj(edge_index_dict)

    neg_candidates = (adj_matrix == 0).nonzero(as_tuple=False)

    # Randomly select a subset of these as negative samples
    torch.manual_seed(seed)

    selected_indices = torch.randperm(neg_candidates.size(0))[:num_neg_samples]
    neg_samples = neg_candidates[selected_indices]

    return neg_samples


def train():

    model.train()
    optimizer.zero_grad()
    
    # %% negative sampling (example code)

    z_encoded = model.encode(train_data.x_dict,
                             train_data.edge_index_dict,
                             0)  # "train_data['Drug', 'Prot'].edge_label_index" not used

    num_prot = train_data['Prot']['num_nodes']
    num_drug = train_data['Drug']['num_nodes']

    train_drug_prot_1 = train_data['Prot', 'Drug'].edge_label_index.clone()
    train_drug_prot_1[1] = train_drug_prot_1[1] + num_prot

    train_drug_prot_2 = train_data['Drug', 'Prot'].edge_label_index.clone()
    train_drug_prot_2[0] = train_drug_prot_2[0] + num_prot

    train_drug_prot_1_label = train_data['Prot', 'Drug'].edge_label.clone()
    train_drug_prot_2_label = train_data['Drug', 'Prot'].edge_label.clone()

    # DTI + TDI (not include PPI and DDI)
    edge_label_index_combine = torch.cat(
        (train_drug_prot_1, train_drug_prot_2[[1, 0]]), dim=1)
    edge_label_combine = torch.cat(
        (train_drug_prot_1_label, train_drug_prot_2_label), dim=0)

    negative_sampling_rate = 1

    num_negative_samples = round(negative_sampling_rate * (edge_label_index_combine).size()[1])
    # print('Num of negative samples', num_negative_samples)

    neg_edge_index = negative_sampling_function(train_data.edge_index_dict, num_neg_samples=num_negative_samples, seed = 333)

    # DTI + TDI (not include PPI and DDI)
    edge_label_index = torch.cat(
        [edge_label_index_combine,  neg_edge_index.t()],  dim=-1)

    edge_label = torch.cat(
        [edge_label_combine, edge_label_combine.new_zeros(neg_edge_index.size(0))], dim=0)
    # %% Model output

    out = model.decode(z_encoded, edge_label_index,
                       0).view(-1)  # z_encoded has value > 1

    loss = F.binary_cross_entropy_with_logits(out, edge_label.float())

    loss.backward()  # tensor(0.7068, grad_fn=<BinaryCrossEntropyBackward0>)

    optimizer.step()

    return float(loss)


# @torch.no_grad()
#
data_in = test_data
data_in = train_data


def test(data_in):
    model.eval()

    z_encoded_1 = model.encode(data_in.x_dict,
                               data_in.edge_index_dict,
                               0)  # "train_data['Drug', 'Prot'].edge_label_index" not used

    num_prot_1 = data_in['Prot']['num_nodes']
    num_drug_1 = data_in['Drug']['num_nodes']

    train_drug_prot_11 = data_in['Prot', 'Drug'].edge_label_index.clone()
    train_drug_prot_11[1] = train_drug_prot_11[1] + num_prot_1

    train_drug_prot_22 = data_in['Drug', 'Prot'].edge_label_index.clone()
    train_drug_prot_22[0] = train_drug_prot_22[0] + num_prot_1

    train_drug_prot_1_label_1 = data_in['Prot', 'Drug'].edge_label.clone()
    train_drug_prot_2_label_2 = data_in['Drug', 'Prot'].edge_label.clone()
    # train_data['Prot', 'interact', 'Prot'].edge_label_index

    # DTI + TDI (not include PPI and DDI)
    edge_label_index_combine_1 = torch.cat(
        (train_drug_prot_11, train_drug_prot_22[[1, 0]]), dim=1)

    edge_label_combine_1 = torch.cat(
        (train_drug_prot_1_label_1, train_drug_prot_2_label_2), dim=0)

    # z_encoded has value > 1
    out_1 = model.decode(z_encoded_1, edge_label_index_combine_1, 0).view(-1)

    roc_score = roc_auc_score(
        edge_label_combine_1.numpy(), out_1.detach().numpy())

    return roc_score


# %% Grid search

params = {
      
#%% Same setting for search
    'layer': [ 4],
    'epochs': [40],
    'learning_rate':[0.01],
    'hidden_channels' :[  8],

}


best_val_record = []
layer_record = []
learnng_rate_record = []

# %% Hyper-parameter tuning
training_record = []
valid_record = []
test_record = []
best_val_auc_record = []


loss_train_record = []
loss_test_record = []
hyper_hidden_channel = []
hyper_layer = []

for lr in params['learning_rate']:

    for layer in params['layer']:

        for hidden_channels in params['hidden_channels']:

            for epochs in params['epochs']:

                lowest_training_loss = 0
                lowest_test_loss = 0
                best_val_auc = final_test_auc = 0

                model = Model(hidden_channels=hidden_channels, num_layers=layer).to(device)

                with torch.no_grad():  # Initialize lazy modules.
                    model.encoder(train_data.x_dict, train_data.edge_index_dict, 0)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                for epoch in range(1, epochs):

                    loss = train()

                    # train_rmse = test(train_data) # actually "binary cross entropy", not "rmse"

                    val_rmse = test(val_data)

                    test_rmse = test(test_data)

                    # Loss recording
                    training_record.append(loss)

                    # AUC recording
                    valid_record.append(val_rmse)  # valid AUC
                    test_record.append(test_rmse)  # test AUC
                    # best_val_auc_record.append()

                    if epoch <= 1:
                        # lowest_training_loss = train_rmse
                        lowest_test_loss = test_rmse
                    else:

                        if lowest_test_loss >= test_rmse:
                            lowest_test_loss = test_rmse
                    if val_rmse > best_val_auc:
                        best_val_auc = val_rmse
                        final_test_auc = test_rmse
                    print(
                        f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_rmse:.4f}, ' f'Test: {test_rmse:.4f}')

                lowest_test_loss = lowest_test_loss
                lowest_training_loss = lowest_training_loss

                best_val_auc_record.append(best_val_auc)
                loss_train_record.append(lowest_training_loss)  # data1
                loss_test_record.append(lowest_test_loss)  # data2
                hyper_hidden_channel.append(hidden_channels)  # axis
                hyper_layer.append(layer)  # axis


# %% Grid search plot

max_val_auc = max(valid_record)  # Find the maximum value in valid_record

# Find the index of best_val_auc in valid_record
best_val_auc_index = valid_record.index(max_val_auc)

#%% training loss
training_record[best_val_auc_index]

# =============================================================================
# %matplotlib qt
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

x = hyper_hidden_channel
y = hyper_layer
z = best_val_auc_record

colors = cm.rainbow(np.linspace(0, 1, len(z)))
# colors = plt.cm.viridis(np.linspace(1, 0, len(z)))
fig = plt.figure()
ax = fig.add_subplot()
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

# Define specific x-ticks to be displayed
desired_xticks = params['hidden_channels']
ax.set_xticks(desired_xticks)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust labelsize as needed

# Creating the scatter plot
scats = ax.scatter(x, y, s=30, c=z, marker='o', cmap=mcolors.ListedColormap(colors))
colorbar = fig.colorbar(scats)
colorbar.ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust labelsize as needed

plt.grid()
plt.xlabel('Number of Hidden Channels', fontsize=16)
plt.ylabel('Number of Layers', fontsize=16)

plt.show()





index_hyper_par = np.argmax(z)
# print("Max AUC index", index_hyper_par)
print("Max AUC", best_val_auc_record[index_hyper_par])
print("Layer number", y[index_hyper_par])
print("Channel number", x[index_hyper_par])


# %% Optimal Net

del model
model = Model(hidden_channels = x[index_hyper_par],
              num_layers = y[index_hyper_par]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'][0])

training_record = []
valid_record = []
# test_record = []


for epoch in range(1, params['epochs'][0]+10):
    loss = train()
    # train_rmse = test(train_data) # actually "binary cross entropy", not "rmse"
    val_rmse = test(val_data)
    test_rmse = test(test_data)

    print(val_rmse)

    # Loss recording
    training_record.append(loss)

    # AUC recording
    valid_record.append(val_rmse)  # valid AUC

# min_test_record = min(test_record)
max_test_idx = np.argmax(valid_record)
print(max_test_idx)
print(valid_record[max_test_idx])


#%% Plot loss and AUC
plt.plot(training_record, label='Training Loss', linewidth=4)  # Increased line width
plt.plot(valid_record, label='AUC score', linewidth=4)  # Increased line width

plt.xlim(0,params['epochs'][0] + 10)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid()
plt.xlabel('Epochs', fontsize=16)
plt.show()

#%% Stop at max AUC

model = Model(hidden_channels=x[index_hyper_par], 
              num_layers=y[index_hyper_par]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'][0])
for epoch in range(1, max_test_idx + 2):
    loss = train()
    # train_rmse = test(train_data) # actually "binary cross entropy", not "rmse"
    val_rmse = test(val_data)
    # test_rmse = test(test_data)
    print(epoch)
    print(val_rmse)


print("Layer number", y[index_hyper_par])
print("Channel number", x[index_hyper_par])


#%% save model
import torch
import os
import pickle

current_path = os.getcwd()



from torch_geometric.data import Data

#%% Save model
# =============================================================================
# # Create some sample data (optional)
# data = Data(data_in)  # Your sample data
# 
# # Specify the file path
# file_path = 'model_zika.pth'
# 
# # Save the model and data
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'data': data,
# }, file_path)
# 
# # Save the model, graph data, and any additional metadata
# additional_metadata = data_temp.metadata()
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'data': data,
#     'additional_metadata': additional_metadata,
# }, file_path)
# =============================================================================

# Load the saved model and data
file_path = 'model_zika.pth'
checkpoint = torch.load(file_path)

saved_state_dict = torch.load(file_path)


# num_node = 3410 + 659
num_layers = 3
num_layer = num_layers

num_layers_before_merging = 1

hidden_channels = 8

# Modify the keys in the saved state dictionary to include angle brackets
modified_state_dict = {}
for key, value in saved_state_dict.items():
    # Modify the keys to include angle brackets
    modified_key = key.replace('__', '<').replace('_', '>')

    # Store the modified key-value pair
    modified_state_dict[modified_key] = value

loaded_model = Model(hidden_channels=hidden_channels, num_layers=num_layers).to(device)

for key in modified_state_dict:
    new_key = key.replace('<', '__').replace('>', '_')
    loaded_model.state_dict()[new_key] = modified_state_dict[key]
    
model = loaded_model

#%% Predict
data_in = test_data
# data_in = data_temp ### entire dataset

num_prot = data_in['Prot']['num_nodes']
num_drug = data_in['Drug']['num_nodes']

# "train_data['Prot', 'Drug'].edge_index" not used
pred, _ = model(data_in.x_dict, data_in.edge_index_dict, 0)

ground_truth = find_Adj(data_temp.edge_index_dict)








is_symmetric = torch.allclose(pred, pred.T)
print(is_symmetric)


# filter existing edges
mask = ground_truth == 0
mask[0:num_prot, 0:num_prot] = 0
mask[num_prot: num_prot + num_drug, num_prot: num_prot + num_drug] = 0

# Use the mask to find the indices in matrixA where matrixB has value 0
indices = (pred * mask).nonzero()
len(indices)


### Ascending order, get most 5 possible interactions
flattened_indices = torch.argsort((pred * mask).flatten())[-25:]
flattened_indices_np = flattened_indices.cpu().numpy()  
indices = np.unravel_index(flattened_indices, pred.shape)
paired_indices = np.column_stack(indices)
indices = torch.tensor(paired_indices)

### find unique

# prot 0 column, drug 1 column
prot_drug_pred_1 = indices[indices[:, 0] < num_prot]
prot_drug_pred_2 = indices[~(indices[:, 0] < num_prot)][:, [1, 0]]  # prot 0 column, drug 1 column

torch.max(prot_drug_pred_1[:,0])   # prot max 658 correct
torch.min(prot_drug_pred_1[:,1])   # drug min 659 correct

### check mapping index

prot_drug_pred = torch.cat((prot_drug_pred_1, prot_drug_pred_2), dim=0)

prot_drug_pred = torch.unique(prot_drug_pred, dim=0, return_inverse=False)


len(prot_drug_pred)

# %% Read mapping
def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir("/")

PPI_path = 'Data/PPI_dat_frame.csv'

PPI_x, PPI_mapping = load_node_csv(PPI_path, index_col='prot')


DDI_path = 'Data/DDI_dat_frame.csv'
# print(pd.read_csv(DDI_path).head())
DDI_x, DDI_mapping = load_node_csv(DDI_path, index_col='parent_key')


PPI_mapping
DDI_mapping

len(DDI_mapping)

# Find map
prot_select = prot_drug_pred[:,0]
drug_select = prot_drug_pred[:,1]
drug_select = drug_select - len(PPI_mapping)

find_name = []

def search_keys_by_value(dictionary, target_value):
    keys = []
    for key, value in dictionary.items():
        if value == target_value:
            keys.append(key)
    return keys

# Iterate over each row in the tensor

prot_drug_pred[:,1] = drug_select

for row in prot_drug_pred:
    
    value_drug = row[1]
    keys_drug = search_keys_by_value(DDI_mapping, value_drug)
    
    value_prot = row[0]
    keys_prot = search_keys_by_value(PPI_mapping, value_prot)
    
    
    find_name.append([keys_prot, keys_drug])

print(find_name)


#%% Save and reload model
import torch

# =============================================================================
# torch.save(model, 'entire_model.pt')
# =============================================================================


# If you saved the entire model:
model = torch.load('entire_model.pt')


#%% BP: forward with only positive edges

model.eval()  # Set the model to evaluation mode

