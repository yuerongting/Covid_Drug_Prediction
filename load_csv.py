# -*- coding: utf-8 -*-
"""
Created on Tue May  3 06:21:54 2022

@author: yrt05
"""

import os.path as osp
import pyreadr
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer
# import sentence-transformers

from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.transforms import RandomLinkSplit, ToUndirected



#%% Load data

# DTI = np.load("C://Users//yrt05//Desktop//Covid_coference_presentation/DTI_label.npy")
os.chdir("C://Users//yrt05//Desktop//Covid_coference_presentation")

DDI = (pyreadr.read_r('DDI_dat_frame.Rdata')['DDI_dat_frame']) # DDI   340 drugs (from 468 drugs)
PPT = (pyreadr.read_r('PPI_dat_frame.Rdata')['PPI_dat_frame']) # PPI   212 prot (from 495 genes)
DTI = (pyreadr.read_r('DTI_dat_frame.Rdata')['DTI_dat_frame']) # DTI   214 d * 64 t



# url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
# root = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
# extract_zip(download_url(url, root), root)
url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
extract_zip(download_url(url, '.'), '.')
# movie_path = osp.join(root, 'ml-latest-small', 'movies.csv')
# rating_path = osp.join(root, 'ml-latest-small', 'ratings.csv')
movie_path = './ml-latest-small/movies.csv'
rating_path = './ml-latest-small/ratings.csv'



print(pd.read_csv(movie_path).head())
print(pd.read_csv(rating_path).head())


#%% Functions

## "encoders=None": not encode the data
# Just return "mapping"
def load_node_csv(path, index_col, encoders=None, **kwargs):   
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


## "src_index_col": index columns of source nodes
## "dst_index_col": index columns of destination nodes
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


# class SequenceEncoder(object):
#     # The 'SequenceEncoder' encodes raw column strings into embeddings.
#     def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
#         self.device = device
#         self.model = SentenceTransformer(model_name, device=device)

#     @torch.no_grad()
#     def __call__(self, df):
#         x = self.model.encode(df.values, show_progress_bar=True,
#                               convert_to_tensor=True, device=self.device)
#         return x.cpu()


# class GenresEncoder(object):
#     # The 'GenreEncoder' splits the raw column strings by 'sep' and converts
#     # individual elements to categorical labels.
#     def __init__(self, sep='|'):
#         self.sep = sep

#     def __call__(self, df):
#         genres = set(g for col in df.values for g in col.split(self.sep))
#         mapping = {genre: i for i, genre in enumerate(genres)}

#         x = torch.zeros(len(df), len(mapping))
#         for i, col in enumerate(df.values):
#             for genre in col.split(self.sep):
#                 x[i, mapping[genre]] = 1
#         return x


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)





#%% Load data
# ## "user_x": Unique userID
# ## "user_mapping": <dict> type, record index
# user_x, user_mapping = load_node_csv(rating_path, index_col='userId')



# ## "movie_x": samples * feature
# movie_x, movie_mapping = load_node_csv(
#     movie_path, index_col='movieId', 
#     # encoders={
#     #     'title': SequenceEncoder(),
#     #     'genres': GenresEncoder()}
#     )



# ## "src_index_col": index columns of source nodes
# ## "dst_index_col": index columns of destination nodes
# edge_index, edge_label = load_edge_csv(
#     rating_path,
#     src_index_col='userId',
#     src_mapping=user_mapping,
#     dst_index_col='movieId',
#     dst_mapping=movie_mapping,
#     # encoders={'rating': IdentityEncoder(dtype=torch.long)},
# )


#%% My data

PPI_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//PPI_dat_frame.csv'
print(pd.read_csv(PPI_path).head())
PPI_x, PPI_mapping = load_node_csv(PPI_path, index_col='prot')

DDI_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//DDI_dat_frame.csv'
print(pd.read_csv(DDI_path).head())
DDI_x, DDI_mapping = load_node_csv(DDI_path, index_col='parent_key')


DTI_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//DTI_dat_frame.csv'
print(pd.read_csv(DTI_path).head())

DDI_feature = (  pyreadr.read_r("jaccard_sim_468_DDI.Rdata")  ['jaccard_sim_468_DDI']).values # DDI smilarity feature 468 x 468
PPI_feature = (  pyreadr.read_r("protein_seq_similarity.RData")  ['protein_seq_similarity']).values # PPI smilarity feature 495 x 495

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PPI_feature_tensor = torch.tensor(PPI_feature).to(dtype=torch.float32)
DDI_feature_tensor = torch.tensor(DDI_feature).to(dtype=torch.float32)



# import tensorflow as tf
# test = tf.convert_to_tensor(DDI_feature)

# DTI_x, DTI_mapping = load_node_csv(DTI_path, index_col='')
# PPI_x, PPI_mapping = load_node_csv(DTI_path, index_col='parent_key')
# DDI_x, DDI_mapping = load_node_csv(DTI_path, index_col='gene_name')

# path =DTI_path
# src_index_col='gene_name'
# src_mapping=PPI_mapping
# dst_index_col='parent_key'
# dst_mapping=DDI_mapping

# df = pd.read_csv(path)



# src = [src_mapping[index] for index in df[src_index_col]]
# dst = [dst_mapping[index] for index in df[dst_index_col]]
# edge_index = torch.tensor([src, dst])

# edge_attr = None
# if encoders is not None:
#     edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
#     edge_attr = torch.cat(edge_attrs, dim=-1)


# for index in df[dst_index_col]:
#     print(index)
#     dst = dst_mapping[index]



edge_index, edge_label = load_edge_csv(
    DTI_path,
    src_index_col='gene_name',
    src_mapping=PPI_mapping,
    dst_index_col='parent_key',
    dst_mapping=DDI_mapping,
    # encoders={'rating': IdentityEncoder(dtype=torch.long)},
)



#%% Hete graph

data = HeteroData()
data['prot'].num_nodes = len(PPI_mapping)  # protein
data['prot'].x = PPI_feature_tensor



data['drug'].num_nodes = len(DDI_mapping)  # drug
data['drug'].x = DDI_feature_tensor

data['prot', 'inter', 'drug'].edge_index = edge_index
# data['user', 'rates', 'movie'].edge_label = edge_label
print(data)

# We can now convert `data` into an appropriate format for training a
# graph-based machine learning model:

    

# 1. Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
data = ToUndirected()(data)
del data['prot', 'inter', 'drug'].edge_label  # Remove "reverse" label.

# 2. Perform a link-level split into training, validation, and test edges.
transform = RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('prot', 'inter', 'drug')],
    rev_edge_types=[('drug', 'rev_inter', 'prot')],
)



train_data, val_data, test_data = transform(data)
print(train_data)
print(val_data)
print(test_data)



#%% Imbalanced data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--epochs', type=int, default=100)


args = parser.parse_args()


if args.use_weighted_loss:
    weight = torch.bincount(train_data['prot', 'drug'].edge_label)
    weight = weight.max() / weight
else:
    weight = None

def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

#%% Training
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear
#%%
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
#%%
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['prot'][row], z_dict['drug'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)
#%%
class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)    # GNNEncoder =2 * SAGEConv
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')  # to_hetero(GNNEncoder)
        
        
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
    
#%%
model = Model(hidden_channels=32).to(device)    ### model


# Due to lazy initialization, we need to run one model step so the number
# of parameters can be inferred:
with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    
    # train_data_edge_index_vae = train_data['prot', 'inter', 'drug'].edge_index
    # train_data_x = 
    
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['prot', 'drug'].edge_label_index)
    target = train_data['prot', 'drug'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)

#%% original VAE
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index),   self.conv_logstd(x, edge_index)

#%% Changing

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # self.encoder = GNNEncoder( in_channels , out_channels)    # GNNEncoder =2 * SAGEConv
        # self.conv1 = to_hetero(self.encoder, data.metadata(), aggr='sum')  # to_hetero(GNNEncoder)
        
        # self.conv_mu = GCNConv(in_channels,  out_channels)
        self.conv_mu = GNNEncoder( in_channels , out_channels)
        self.conv_logstd = GNNEncoder( in_channels , out_channels)
        
        
        
        
        self.encoder = GNNEncoder(in_channels , out_channels)    # GNNEncoder =2 * SAGEConv
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')  # to_hetero(GNNEncoder)
        
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # x = self.encoder(x, edge_index).relu()
        # return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index), self.decoder(z_dict, edge_label_index)


#%%    
model = VGAE(VariationalGCNEncoder(in_channels, out_channels))     ### model: VGAE(VariationalGCNEncoder)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    
    
    # z = model.encode(train_data.x, train_data.edge_index)
    z = model.encode(train_data.x_dict, train_data.edge_index_dict, train_data["edge_label_index"])
    
    loss = model.recon_loss(z, train_data.edge_label_dict)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)





# import numpy as np
# drug_num = DDI_feature.shape[0]
# prot_num = PPI_feature.shape[0]
# data_x = np.zeros([drug_num * prot_num , drug_num + prot_num])   ###   258*9 x (258+9*2)
# flag = 0
# for i in range(drug_num):  # drug 0
#     for j in range(prot_num):  # prot 0 , 1 , ... 9
#         data_x[flag] = np.concatenate(( DDI_feature[i,] , PPI_feature[j,:]))
#         flag = flag + 1

# data_x.shape
# X = data_x

# z_dict = model.encoder(train_data.x_dict, train_data.edge_label_index_dict )

# z = torch.cat([   train_data.x_dict['prot']  , train_data.x_dict['drug']  ], dim=0)



# row, col = train_data.edge_index_dict
# test = torch.cat([train_data.x_dict['prot'][row], train_data.x_dict['drug'][col]], dim=-1)




import torch.nn.functional as F

@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['prot', 'drug'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['prot', 'drug'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


for epoch in range(1, 301):
    loss = train()
    train_rmse = test(train_data)
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')

#%%
# class HeteroGNN(torch.nn.Module):
#     def __init__(self, metadata, hidden_channels, out_channels, num_layers):
#         super().__init__()

#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             conv = HeteroConv({
#                 edge_type: SAGEConv((-1, -1), hidden_channels)
#                 for edge_type in metadata[1]
#             })
#             self.convs.append(conv)

#         self.lin = Linear(hidden_channels, out_channels)

#     def forward(self, x_dict, edge_index_dict):
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
#         return self.lin(x_dict['author'])


# model = HeteroGNN(data.metadata(), hidden_channels=64, out_channels=4,
#                   num_layers=2)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data, model = data.to(device), model.to(device)

# with torch.no_grad():  # Initialize lazy modules.
#     out = model(data.x_dict, data.edge_index_dict)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x_dict, data.edge_index_dict)
#     mask = data['author'].train_mask
#     loss = F.cross_entropy(out[mask], data['author'].y[mask])
#     loss.backward()
#     optimizer.step()
#     return float(loss)


# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

#     accs = []
#     for split in ['train_mask', 'val_mask', 'test_mask']:
#         mask = data['author'][split]
#         acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
#         accs.append(float(acc))
#     return accs


# for epoch in range(1, 101):
#     loss = train()
#     train_acc, val_acc, test_acc = test()
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
#           f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')



#%%  Auto encoder


from torch_geometric.nn import GAE, VGAE, GCNConv


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
parser.add_argument('--variational', action='store_true', default = True)
parser.add_argument('--linear', action='store_true')
parser.add_argument('--epochs', type=int, default=100)


args = parser.parse_args()



class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)




in_channels, out_channels = 32, 16


select = True
if(select):
    if not args.variational and not args.linear:
        model = GAE(GCNEncoder(in_channels, out_channels)) 
    elif not args.variational and args.linear:
        model = GAE(LinearEncoder(in_channels, out_channels))
        
    elif args.variational and not args.linear:                          ## use this GAE, no linear encoder
        model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
        
    elif args.variational and args.linear:
        model = VGAE(VariationalLinearEncoder(in_channels, out_channels))


model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x_dict, train_data.edge_index_dict)
    
    
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

# train_data.x_dict, train_data.edge_index_dict
@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x_dict, data.edge_index_dict)
    # train_data.x_dict, train_data.edge_index_dict
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


for epoch in range(1, args.epochs + 1):
    loss = train()
    auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    
    
    # pred = model(train_data.x_dict, train_data.edge_index_dict,
    #              train_data['prot', 'drug'].edge_label_index)
    



#%% variational data example
import argparse
import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=400)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
path = ''
dataset = Planetoid(path, args.dataset, transform=transform)
train_data, val_data, test_data = dataset[0]

train_data_x = train_data.x
train_data_edge_index = train_data.edge_index
