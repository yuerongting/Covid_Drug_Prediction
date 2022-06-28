# -*- coding: utf-8 -*-
"""
Created on Tue May  3 06:21:54 2022

@author: yrt05
"""
import numpy as np
import os.path as osp
import pyreadr
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer
# import sentence-transformers

from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear
import torch.nn.functional as F
# from torch_geometric.nn import DenseSAGEConv
# from torch_geometric.nn.dense import DenseGraphConv


#%% Example data
# # url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
# # root = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
# # extract_zip(download_url(url, root), root)
# url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
# extract_zip(download_url(url, '.'), '.')
# # movie_path = osp.join(root, 'ml-latest-small', 'movies.csv')
# # rating_path = osp.join(root, 'ml-latest-small', 'ratings.csv')
# movie_path = './ml-latest-small/movies.csv'
# rating_path = './ml-latest-small/ratings.csv'



# print(pd.read_csv(movie_path).head())
# print(pd.read_csv(rating_path).head())

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


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)






#%% My data


# DTI = np.load("C://Users//yrt05//Desktop//Covid_coference_presentation/DTI_label.npy")
os.chdir("C://Users//yrt05//Desktop//Covid_coference_presentation")
# cd C://Users//yrt05//Desktop//Covid_coference_presentation


### test = pyreadr.read_r('DDI_dat_frame.Rdata')['DDI_dat_frame']


DDI = (pyreadr.read_r('DDI_dat_frame.Rdata')['DDI_dat_frame']) # DDI   340 drugs (from 468 drugs)
PPT = (pyreadr.read_r('PPI_dat_frame.Rdata')['PPI_dat_frame']) # PPI   212 prot (from 495 genes)
DTI = (pyreadr.read_r('DTI_dat_frame.Rdata')['DTI_dat_frame']) # DTI   214 d * 64 t

PPI_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//PPI_dat_frame.csv'
print(pd.read_csv(PPI_path).head())
PPI_x, PPI_mapping = load_node_csv(PPI_path, index_col='prot')

DDI_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//DDI_dat_frame.csv'
print(pd.read_csv(DDI_path).head())
DDI_x, DDI_mapping = load_node_csv(DDI_path, index_col='parent_key')




DDI_feature = (  pyreadr.read_r("jaccard_sim_468_DDI.Rdata")  ['jaccard_sim_468_DDI']).values # DDI smilarity feature 468 x 468
PPI_feature = (  pyreadr.read_r("protein_seq_similarity.RData")  ['protein_seq_similarity']).values # PPI smilarity feature 495 x 495

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PPI_feature_tensor = torch.tensor(PPI_feature).to(dtype=torch.float32)
DDI_feature_tensor = torch.tensor(DDI_feature).to(dtype=torch.float32)

len(PPI_mapping)
len(PPI_feature_tensor)
len(DDI_feature_tensor)



drug_prot_RWR = (  pyreadr.read_r("drug_prot_RWR.Rdata")  ['drug_prot_RWR']).values # RWR = drug + prot

adj_all = (  pyreadr.read_r("adj_all.RData")  ['adj_all']).values # adj_all = drug * target

DT_similarity = np.dot(np.dot(DDI_feature , adj_all), PPI_feature)  ### dt_similar = drug * target

# DT_similarity.size()
# np.size(DT_similarity)
# dim(DT_similarity)

#%% edge_index

# DTI_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//DTI_dat_frame.csv'
# print(pd.read_csv(DTI_path).head())

# edge_index, edge_label = load_edge_csv(
#     DTI_path,
#     src_index_col='gene_name',
#     src_mapping=PPI_mapping,
#     dst_index_col='parent_key',
#     dst_mapping=DDI_mapping,
#     # encoders={'rating': IdentityEncoder(dtype=torch.long)},
# )

All_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//edge_all.csv'
print(pd.read_csv(All_path).head())



DDI_mapping_add_PPI = {key_: val_+len(DDI_mapping)  for key_, val_ in PPI_mapping.items()} 
# All_mapping = {**PPI_mapping, **DDI_mapping_add_PPI}  ### prot + drug
All_mapping = { **DDI_mapping, **DDI_mapping_add_PPI}  ### drug + prot

edge_index, edge_label = load_edge_csv(
    All_path,
    src_index_col='V1',
    src_mapping=All_mapping,
    dst_index_col='V2',
    dst_mapping=All_mapping,
    # encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

edge_index.size()   ### 51700 edges

np.where(edge_index[0]<600)



# #%% Edge weight

# edge_weight = (  pyreadr.read_r("edge_weight.Rdata")  ['edge_weight']).values # PPI smilarity feature 495 x 495




# #%% Regular net for "edge_index"
# # from torch_geometric.data import Data
# # data = Data(edge_index=edge_index, num_nodes= len(PPI_mapping) + len(DDI_mapping))

# # from torch_geometric.transforms import AddSelfLoops
# # assert AddSelfLoops().__repr__() == 'AddSelfLoops()'
# # data = AddSelfLoops()(data)

# #%% Hete graph

# data = HeteroData()
# # data = Data()
# data['prot'].num_nodes = len(PPI_mapping)  # protein
# data['prot'].x = PPI_feature_tensor

# data['drug'].num_nodes = len(DDI_mapping)  # drug
# data['drug'].x = DDI_feature_tensor

# # data['prot', 'inter', 'drug'].edge_index = edge_index
# data['prot', 'drug'].edge_index = edge_index
# # data['prot', 'drug'].edge_weight = torch.randn(len(edge_index[0]), )
# data['prot', 'drug'].edge_weight = edge_weight

# # data['user', 'rates', 'movie'].edge_label = edge_label


# # 1. Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
# data = ToUndirected()(data)
# del data['prot',  'drug'].edge_label  # Remove "reverse" label.

# ###%% Self-loop
# # from torch_geometric.transforms import AddSelfLoops
# # data = AddSelfLoops()(data)
# # data = AddSelfLoops(attr='edge_weight', fill_value=5)(data)
# # assert data.edge_weight.tolist() == [1, 2, 3, 4, 5, 5, 5]


# # 2. Perform a link-level split into training, validation, and test edges.
# transform = RandomLinkSplit(
#     num_val=0.1,
#     num_test=0.1,
#     neg_sampling_ratio=0.9,
#     edge_types=[('prot', 'drug')],
#     rev_edge_types=[('drug',  'prot')],
# )


# train_data, val_data, test_data = transform(data)
# # train_data.pos_edge_label_index
# # data['prot', 'drug'].edge_label.float()

# # train_data['prot', 'drug'].edge_index
# test_data['prot', 'drug'].edge_index[:,1:10]

# # test_dataset, val_dataset, train_dataset = transform(data)








# #%% Model Hetero 

# class GNNEncoder(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels):
#         super().__init__()
#         # in_channels = 16
        
#         # self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize=True)
#         # self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        
#         self.conv1 = SAGEConv((-1, -1), hidden_channels)
#         self.conv2 = SAGEConv((-1, -1), out_channels)
#         # self.conv1 = GCNConv(in_channels, hidden_channels, cached = True)
#         # self.conv2 = GCNConv(hidden_channels, out_channels, cached = True)
        
        
        
#     def bn(self, i, x):
#         batch_size, num_nodes, num_channels = x.size()

#         x = x.view(-1, num_channels)
#         x = getattr(self, f'bn{i}')(x)
#         x = x.view(batch_size, num_nodes, num_channels)
#         return x
    
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index)
#         return x
#     # def forward(self, x, adj, mask=None):
#     #     batch_size, num_nodes, in_channels = x.size()

#     #     x0 = x
#     #     x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
#     #     # x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
#     #     # x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

#     #     # x = torch.cat([x1, x2, x3], dim=-1)
#     #     x = x1

#     #     if self.lin is not None:
#     #         x = F.relu(self.lin(x))

#     #     return x
    
# class EdgeDecoder(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.lin1 = Linear(2 * hidden_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, 1)

#     def forward(self, z_dict, edge_label_index):
#         row, col = edge_label_index
#         z = torch.cat([z_dict['prot'][row], z_dict['drug'][col]], dim=-1)

#         z = self.lin1(z).relu()
#         z = self.lin2(z)
#         return z.view(-1)

# class Model(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.encoder = GNNEncoder(hidden_channels, hidden_channels)    # GNNEncoder =2 * SAGEConv
#         self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')  # to_hetero(GNNEncoder)
        
        
#         self.decoder = EdgeDecoder(hidden_channels)

#     def forward(self, x_dict, edge_index_dict, edge_label_index):
#         z_dict = self.encoder(x_dict, edge_index_dict)
#         return self.decoder(z_dict, edge_label_index)
    
# model = Model(hidden_channels=32).to(device)    ### model

# with torch.no_grad():
#     model.encoder(train_data.x_dict, train_data.edge_index_dict)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



# ###  Imbalanced data

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--use_weighted_loss', action='store_true', help='Whether to use weighted MSE loss.', default=True)
# parser.add_argument('--variational', action='store_true')
# parser.add_argument('--linear', action='store_true')
# parser.add_argument('--epochs', type=int, default=100)
# args = parser.parse_args()

# if args.use_weighted_loss:
#     weight = torch.bincount((train_data['prot', 'drug'].edge_label).int())
#     weight = weight.max() / weight
# else:
#     weight = None

# def weighted_mse_loss(pred, target, weight=None):
#     weight = 1. if weight is None else weight[target.long()].to(pred.dtype)
#     return (weight *   (pred - target.to(pred.dtype)).pow(2)  ).mean()




# def train():
#     model.train()
#     optimizer.zero_grad()
    
#     # train_data_edge_index_vae = train_data['prot', 'drug'].edge_index
#     # train_data_x = 
    
#     pred = model(train_data.x_dict, train_data.edge_index_dict,
#                  train_data['prot', 'drug'].edge_label_index)
#     target = train_data['prot', 'drug'].edge_label
#     loss = weighted_mse_loss(pred, target, weight)
#     loss.backward()
#     optimizer.step()
#     return float(loss)

# @torch.no_grad()
# def test(data):
#     model.eval()
#     pred = model(data.x_dict, data.edge_index_dict,
#                  data['prot', 'drug'].edge_label_index)
#     pred = pred.clamp(min=0, max=1)
#     target = data['prot', 'drug'].edge_label.float()
#     rmse = F.mse_loss(pred, target).sqrt()
#     return float(rmse)


# # get label of "pred, target"
# def test_val(data):
#     model.eval()
#     pred = model(data.x_dict, data.edge_index_dict,
#                  data['prot', 'drug'].edge_label_index)
#     pred = pred.clamp(min=0, max=1)
#     target = data['prot', 'drug'].edge_label.float()
#     rmse = F.mse_loss(pred, target).sqrt()
#     return float(rmse), pred, target

# for epoch in range(1, 10):
#     loss = train()
#     train_rmse = test(train_data)
#     val_rmse = test(val_data)
#     test_rmse, test_pred, test_target = test_val(test_data)
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
#           f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')


# test_pred[test_pred<0.5] = 0
# test_pred[test_pred>0.5] = 1

# test_pred
# test_target

# # print(f'test_pred: {test_pred:03d}, test_target: {test_target:.4f}')







#%% Link prediction:  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py

# import os.path as osp

import torch
from sklearn.metrics import roc_auc_score

# import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import tensorflow as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transform = T.Compose([
#     T.NormalizeFeatures(),
#     T.ToDevice(device),
#     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#                       add_negative_train_samples=False),
# ])
# path = '..\data\Planetoid'
# dataset = Planetoid(path, name='Cora', transform=transform)


# # After applying the `RandomLinkSplit` transform, the data is transformed from
# # a data object to a list of tuples (train_data, val_data, test_data), with
# # each element representing the corresponding split.
# train_data, val_data, test_data = dataset[0]
# test = train_data.x
# test1 = data.x 



# edge_index, edge_label = load_edge_csv(
#     DTI_path,
#     src_index_col='gene_name',
#     src_mapping=PPI_mapping,
#     dst_index_col='parent_key',
#     dst_mapping=DDI_mapping,
#     # encoders={'rating': IdentityEncoder(dtype=torch.long)},
# )

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    neg_sampling_ratio=1.0,
    # edge_types=[('prot', 'drug')],
    # rev_edge_types=[('drug',  'prot')],
)




# edge_index[1] = edge_index[1] + len(PPI_mapping)  ### edge_index = protein + drug
data = Data( edge_index=edge_index, num_nodes= len(All_mapping))

# data.edge_index[0]

# np.where((data.edge_index)>800)
# np.where((data.edge_index)<100)
# num_features = 10   ### Features on nodes

feature_row1 = np.hstack((DDI_feature, DT_similarity))
feature_row2 = np.hstack((DT_similarity.T, PPI_feature))
feature = np.vstack((feature_row1, feature_row2))



#%% PCA
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pca_1 = PCA(svd_solver = 'full',n_components = 10)
pipe_1 = Pipeline([('scaler', StandardScaler()), ('pca', pca_1)])

X = feature
X_transformed = pipe_1.fit_transform(X)

X_transformed.shape

main = pca_1.explained_variance_ratio_
plt.style.use('seaborn')
plt.plot(main,'o--')
# plt.title('PCA--explained variance ratio verses number of components')
plt.xlabel('Principal Components')
plt.ylabel('Percentage of Explained Variance')



pca_2 = PCA(svd_solver = 'full',n_components = 2)
pipe_2 = Pipeline([('scaler', StandardScaler()), ('pca', pca_2)])
X = feature
data_feature = pipe_2.fit_transform(X)



#%% Test   VGAE


# from torch_geometric.nn import SGConv
# from torch_geometric.nn import GAE, VGAE, GCNConv
# class VariationalGCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv_mu = GCNConv(in_channels, out_channels)
#         self.conv_logstd = GCNConv(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# model = VGAE(VariationalGCNEncoder(num_features, 8, 2)).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# def train():
#     model.train()
#     optimizer.zero_grad()
#     z = model.encode(train_data.x, train_data.edge_index)
    
#     neg_edge_index = negative_sampling(
#         edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
#         num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    
    
#     pos_edge_label_index = train_data.edge_index
    
#     edge_label_index = torch.cat(
#         [train_data.edge_label_index, neg_edge_index],
#         dim=-1,
#     )
#     # edge_label = torch.cat([
#     #     train_data.edge_label,
#     #     train_data.edge_label.new_zeros(neg_edge_index.size(1))
#     # ], dim=0)
    
    
#     # z = model.encode(train_data.x, edge_label_index)
    
#     # loss = model.recon_loss(z, edge_label_index)
#     loss = model.recon_loss(z, pos_edge_label_index)
    
#     variational = True
#     if variational:
#         loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
#     loss.backward()
#     optimizer.step()
#     return float(loss)


# @torch.no_grad()
# def test(data):
#     model.eval()
#     z = model.encode(data.x, data.edge_index)
#     # return model.test(z, data.edge_label_index, data.neg_edge_index)
#     return model.test(z, data.edge_label_index, data.edge_label_index)

# epoch = 100
# for epoch in range(1, epoch):
#     loss = train()
#     auc, ap = test(test_data)
#     print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    
#%% Training



data.x = torch.tensor(             data_feature                ).to(dtype=torch.float32) 

# edge_weight = drug_prot_RWR[:,1].astype(float)
# edge_weight =  torch.from_numpy(            edge_weight           )

# data.edge_weight = edge_weight



train_data, val_data, test_data = transform(data)
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached = True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached = True)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
num_features = data_feature.shape[1]

model = Net(num_features, 8, 2).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy()), out

val_auc_record = []
test_auc_record = []
loss_record = []


epoch = 100
best_val_auc = final_test_auc = 0
for epoch in range(1, epoch):
    loss = train()
    val_auc,val_out = test(val_data)
    test_auc,test_out = test(test_data)
    
    test_auc_record.append(test_auc)
    loss_record.append(loss)
    val_auc_record.append(val_auc)
    
    if val_auc > best_val_auc:
        best_val = val_auc
        final_test_auc = test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')





epo = np.linspace(1, epoch,epoch)
plt.style.use('seaborn')
plt.plot(epo, test_auc_record, 'r', label = 'test_auc_record')
plt.plot(epo, torch.Tensor(loss_record)/  (max(loss_record).detach().numpy()) , 'b', label = 'loss_record')
plt.plot(epo, val_auc_record, 'g', label = 'val_auc_record')
plt.xlabel('Epoch')
plt.legend()



z = model.encode(test_data.x, test_data.edge_index)
final_edge_index = model.decode_all(z)
final_edge_index.size()

edge_final = final_edge_index.detach().numpy()




#%% Find DTI edges


test = pd.DataFrame(edge_final.T, columns= ['V1', 'V2'])

# test.V1 == All_mapping.values()
# (All_mapping.values())[0]

test_no_dup = test[test.V1 != test.V2]

# test.drop_duplicates(keep=False,inplace=True)



All_mapping = { **DDI_mapping, **DDI_mapping_add_PPI}  ### drug + prot

len(DDI_mapping)

edge_DTI_pred = test_no_dup[(test_no_dup.V1 < len(DDI_mapping)) & (test_no_dup.V2 >= len(DDI_mapping))]

# test_no_dup[(test_no_dup.V2 < len(DDI_mapping)) & (test_no_dup.V1 >= len(DDI_mapping))]  ### same number, symmetric

edge_DTI_pred





listOfDrug = list()
for i in edge_DTI_pred.V1:
    ind_drug =  list(All_mapping.keys())[list(All_mapping.values()).index(i)]
    listOfDrug.append(ind_drug)

listOfProt = list()
for j in edge_DTI_pred.V2:
    ind_prot =  list(All_mapping.keys())[list(All_mapping.values()).index(j)]
    listOfProt.append(ind_prot)


df = pd.DataFrame(list(zip(listOfDrug, listOfProt)), columns =['Drug', 'Prot']) 

df.to_excel (r'drug_prot.xlsx', index = False, header=True)















#%% variational data example
# import argparse
# import os.path as osp

# import torch
# import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
# from torch_geometric.nn import GAE, VGAE, GCNConv

# parser = argparse.ArgumentParser()
# parser.add_argument('--variational', action='store_true')
# parser.add_argument('--linear', action='store_true')
# parser.add_argument('--dataset', type=str, default='Cora',
#                     choices=['Cora', 'CiteSeer', 'PubMed'])
# parser.add_argument('--epochs', type=int, default=400)
# args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform = T.Compose([
#     T.NormalizeFeatures(),
#     T.ToDevice(device),
#     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#                       split_labels=True, add_negative_train_samples=False),
# ])
# # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# path = ''
# dataset = Planetoid(path, args.dataset, transform=transform)
# train_data, val_data, test_data = dataset[0]

# train_data_x = train_data.x  # 
# train_data_edge_index = train_data.edge_index
# train_data_pos_edge_label_index = train_data.pos_edge_label_index
