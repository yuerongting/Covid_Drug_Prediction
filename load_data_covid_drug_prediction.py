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
import fsspec
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.nn import SAGEConv, to_hetero, GraphConv
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.nn import Parameter
from torch_geometric.nn import SGConv
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.nn import Sequential, HeteroConv
import random
random.seed(1)
torch.use_deterministic_algorithms(True)
np.random.seed(1)        
torch.manual_seed(1)
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#%% Functions

# ## "encoders=None": not encode the data
# # Just return "mapping"
def load_node_csv(path, index_col, encoders=None, **kwargs):   
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


# ## "src_index_col": index columns of source nodes
# ## "dst_index_col": index columns of destination nodes
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


#%% My data
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.chdir("C://Users//yrt05//Desktop//Covid_coference_presentation")
# cd C://Users//yrt05//Desktop//Covid_coference_presentation

# Bind_data = pd.read_csv('BindingDB_All.tsv', sep='\t')



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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# PPI_feature_tensor = torch.tensor(PPI_feature).to(dtype=torch.float32)
# DDI_feature_tensor = torch.tensor(DDI_feature).to(dtype=torch.float32)

# len(PPI_mapping)
# len(PPI_feature_tensor)
# len(DDI_feature_tensor)



drug_prot_RWR = (  pyreadr.read_r("drug_prot_RWR.Rdata")  ['drug_prot_RWR']).values # RWR = drug + prot



#%% PCA for Drug
x_drug = (  pyreadr.read_r("x_drug.Rdata")  ['x']).values 
print(x_drug.shape)

x_drug = np.concatenate((x_drug, DDI_feature), axis = 1)  # concat:  x_drug, DDI_feature
print(x_drug.shape)

# num_PCA = int(x_drug.shape[1]*2/3)
num_PCA_drug = 20

# num_take = 10
pca_1_drug = PCA(svd_solver = 'full',n_components = num_PCA_drug) # show first 10 PCs
pipe_1 = Pipeline([('scaler', StandardScaler()), ('pca', pca_1_drug)])

x_drug.shape

X = x_drug
# X = x_prot

X_transformed = pipe_1.fit_transform(X)

# X_transformed.shape

main_drug = pca_1_drug.explained_variance_ratio_



sum(main_drug[0:20])


# # fig, ax = plt.subplots()
# plt.style.use('default')
# plt.plot(main,'o--', label = 'Explained Variance of each PC')
# # plt.axis(fontsize = 18)
# plt.xlabel('Principal Components', fontsize = 18)
# plt.ylabel('Explained Variance of Drug Data', fontsize = 18)
# plt.grid()
# plt.bar(range(0,num_PCA), pca_1.explained_variance_ratio_,
#         alpha=0.5,
#         align='center')
# plt.step(range(0,num_PCA), np.cumsum(pca_1.explained_variance_ratio_),
#          where='mid',
#          color='red', label = 'Accumulative Explained Variance')
# # plt.legend(loc = 'center right')

# plt.show()

x_drug = X_transformed



#%% PCA for Prot

PPI_feature.shape

x_prot = (  pyreadr.read_r("x_prot.RData")  ['prot_property']).values 
print(x_prot.shape)

x_prot = np.concatenate((x_prot, PPI_feature), axis = 1)  # concat:  x_prot, PPI_feature

print(x_prot.shape)

# num_PCA = int(x_prot.shape[1]*2/3)
# num_PCA = 350
num_PCA = 300

# num_take = 10
pca_1 = PCA(svd_solver = 'full',n_components = num_PCA) # show first 10 PCs
pipe_1 = Pipeline([('scaler', StandardScaler()), ('pca', pca_1)])
x_prot.shape
X = x_prot

X_transformed = pipe_1.fit_transform(X)

# X_transformed.shape

main = pca_1.explained_variance_ratio_

sum(main[0:300])

np.sum(main)
plt.style.use('default')


x_prot = X_transformed




#%% PCA plots
fig = plt.figure()
# fig.tight_layout() 
plt.subplots_adjust(hspace= 1)
plt.subplot(221)
plt.plot(main_drug,'o--', label = 'Explained Variance of each PC')
plt.xlabel('PCs', fontsize = 18)
plt.ylabel('Variance', fontsize = 18)
plt.grid()
plt.bar(range(0,num_PCA_drug), pca_1_drug.explained_variance_ratio_,
        alpha=0.5,
        align='center')
plt.step(range(0,num_PCA_drug), np.cumsum(pca_1_drug.explained_variance_ratio_),
         where='mid',
         color='red', label = 'Accumulative Explained Variance')
plt.title('(b) Drug Data', fontsize = 16, y=-1.5)
plt.legend(loc = 'best', bbox_to_anchor =(2, 2), fontsize = 16 )
# plt.show()





ax = plt.subplot(222)

plt.plot(main,'o--', label = 'Explained Variance of each PC')
# plt.axis(fontsize = 18)
plt.xlabel('PCs', fontsize = 18)
# plt.ylabel('Variance', fontsize = 18)
plt.grid()
plt.bar(range(0,num_PCA), pca_1.explained_variance_ratio_,
        alpha=0.5,
        align='center')
plt.step(range(0,num_PCA), np.cumsum(pca_1.explained_variance_ratio_),
         where='mid',
         color='red', label = 'Accumulative Explained Variance')
plt.title('(c) Protein Data', fontsize = 16, y=-1.5)
# plt.legend(loc = 'center right')

# ax.legend(bbox_to_anchor=(1.1, 1.05), fontsize = 16)
# fig.tight_layout(pad=5.0)
plt.show()




#%% Hete graph

class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


# All_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//edge_all.csv'
All_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//edge_all_label.csv'

def load_edge_csv_part(path, src_index_col, src_mapping, dst_index_col, dst_mapping,begin, end, 
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)[begin:end]

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


# all_edge_type = c("DDI", "PPI", "DTI")

edge_index_ddi, edge_label_ddi = load_edge_csv_part(
    All_path,
    src_index_col='V1',
    src_mapping=DDI_mapping,
    dst_index_col='V2',
    dst_mapping=DDI_mapping,
    begin = 0,
    end = 48460,
    # encoders={'label_type': IdentityEncoder(dtype=torch.long)}
    # encoders={'edge_type': IdentityEncoder(dtype=torch.long)},
)

edge_index_ppi, edge_label_ppi = load_edge_csv_part(
    All_path,
    src_index_col='V1',
    src_mapping=PPI_mapping,
    dst_index_col='V2',
    dst_mapping=PPI_mapping,
    begin = 48460,
    end = 51405, 
    # encoders={'label_type': IdentityEncoder(dtype=torch.long)},
    # encoders={'edge_type': IdentityEncoder(dtype=torch.long)},
)

edge_index_dti, edge_label_dti = load_edge_csv_part(
    All_path,
    src_index_col='V1',
    src_mapping=DDI_mapping,
    dst_index_col='V2',
    dst_mapping=PPI_mapping,
    begin = 51405,
    end = 51700,
    # encoders={'label_type': IdentityEncoder(dtype=torch.long)},
    # encoders={'edge_type': IdentityEncoder(dtype=torch.long)},
)

#%% Test edge label
edge_index_ddi, edge_label_ddi = load_edge_csv_part(
    All_path,
    src_index_col='V1',
    src_mapping=DDI_mapping,
    dst_index_col='V2',
    dst_mapping=DDI_mapping,
    begin = 0,
    end = 48460,
    # encoders={'label_type': IdentityEncoder(dtype=torch.long)}
    encoders={'edge_type': IdentityEncoder(dtype=torch.long)},
)

edge_index_ppi, edge_label_ppi = load_edge_csv_part(
    All_path,
    src_index_col='V1',
    src_mapping=PPI_mapping,
    dst_index_col='V2',
    dst_mapping=PPI_mapping,
    begin = 48460,
    end = 51405, 
    # encoders={'label_type': IdentityEncoder(dtype=torch.long)},
    encoders={'edge_type': IdentityEncoder(dtype=torch.long)},
)

edge_index_dti, edge_label_dti = load_edge_csv_part(
    All_path,
    src_index_col='V1',
    src_mapping=DDI_mapping,
    dst_index_col='V2',
    dst_mapping=PPI_mapping,
    begin = 51405,
    end = 51700,
    # encoders={'label_type': IdentityEncoder(dtype=torch.long)},
    encoders={'edge_type': IdentityEncoder(dtype=torch.long)},
)

all_edge = torch.cat([edge_index_ddi, edge_index_ppi, edge_index_dti], dim = 1)
all_edge_label = torch.cat([edge_label_ddi, edge_label_ppi, edge_label_dti], dim = 0)


edge_index = all_edge

### Node
# row, col = edge_index
# node_local = x[row] # node mean aggregate other nodes 
# node_agg = scatter_mean(node_local, col, dim=0, dim_size=x.size(0)) # Mean aggregation
# # node_agg.shape # 75 nodes in total

# # edge_index.shape
# # edge_attr.shape
edge_attr = torch.ones([len(edge_index[0])])

for i in range(len(all_edge_label)):
    for j in all_edge_label:
        if all_edge_label[i] == j:
            edge_attr[i] = torch.tensor([0,1])
##################################################################################################################

### Edge: edge_agg      
edge_agg = torch.zeros(len(edge_index[0]), number_of_edge_features)

len(edge_index[0])
i = 1

for i in range(len(edge_index[0])):
    src, dst = edge_index[:,i] # index for target edge
    
    ### for each edge "i j", mean aggregate its neighbors
    edge_neighbor_mean = torch.mean(torch.cat([ edge_attr[ (row == src) & (~((row == src) & (col == dst))) ]   , edge_attr[col == dst],  ], dim = 0), dim = 0)
    edge_agg[i] = edge_neighbor_mean

#%%



data = HeteroData()
data['Prot'].num_nodes = len(x_prot) 
data['Prot'].x = torch.from_numpy( x_prot).float()
data['Drug'].num_nodes = len(x_drug) 
data['Drug'].x = torch.from_numpy( x_drug).float()


data['Prot', 'interact', 'Prot'].edge_index = edge_index_ppi
# data['Prot', 'interact', 'Prot'].edge_label = edge_label_ppi

data['Drug', 'interact', 'Drug'].edge_index = edge_index_ddi

data['Drug', 'bind', 'Prot'].edge_index = edge_index_dti

# ### Edge label
# edge_type_label = []
# label = 0
# for edge_type in metadata[1]:
#     # data[edge_type].edge_index.shape
#     edge_type_label.append( (torch.ones( data[edge_type].edge_index.shape[1] )) * label  )
#     label = label + 1

# # edge_type_label



data = ToUndirected(merge=False)(data)

metadata = data.metadata()

# ### Cross validation
# def k_fold(dataset, folds):
#     skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

#     test_indices, train_indices = [], []
#     for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
#         test_indices.append(torch.from_numpy(idx).to(torch.long))

#     val_indices = [test_indices[i - 1] for i in range(folds)]

#     for i in range(folds):
#         train_mask = torch.ones(len(dataset), dtype=torch.bool)
#         train_mask[test_indices[i]] = 0
#         train_mask[val_indices[i]] = 0
#         train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

#     return train_indices, test_indices, val_indices



# from sklearn.model_selection import StratifiedKFold
# folds = 5
# dataset = data
# k_fold(data, folds = folds)


# for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):

#         train_dataset = dataset[train_idx]
#         test_dataset = dataset[test_idx]
#         val_dataset = dataset[val_idx]

#         if 'adj' in train_dataset[0]:
#             train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
#             val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
#             test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
#         else:
#             train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
#             val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
#             test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


train_data, val_data, test_data = transform = RandomLinkSplit(is_undirected=True,edge_types=data.edge_types,)(data)





















#%%
#%%
#%%
#%%
#%% Graph data

import torch
from sklearn.metrics import roc_auc_score

from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import tensorflow as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    neg_sampling_ratio=1.0,
)

# No split, only generate "edge_label_index"
transform2 = RandomLinkSplit(
    num_val=0,
    num_test=1,
    neg_sampling_ratio=0,
    # edge_types=[('prot', 'drug')],
    # rev_edge_types=[('drug',  'prot')],
)


data = Data( edge_index=edge_index, num_nodes= len(All_mapping))





#%% PCA
num_PCA = 150


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pca_1 = PCA(svd_solver = 'full',n_components = num_PCA) # show first 10 PCs
pipe_1 = Pipeline([('scaler', StandardScaler()), ('pca', pca_1)])


feature = x_drug # drug PCA
feature = x_prot # prot PCA


X = feature
X_transformed = pipe_1.fit_transform(X)

X_transformed.shape

main = pca_1.explained_variance_ratio_

np.sum(main)

plt.style.use('default')
plt.plot(main,'o--')
plt.xlabel('Principal Components', fontsize = 18)
plt.ylabel('Explained Variance', fontsize = 18)
plt.grid()


plt.bar(range(0,num_PCA), pca_1.explained_variance_ratio_,
        alpha=0.5,
        align='center')
plt.step(range(0,num_PCA), np.cumsum(pca_1.explained_variance_ratio_),
         where='mid',
         color='red')

plt.show()





#%% Data features
data.x = torch.tensor(             data_feature                ).to(dtype=torch.float32) 
data.x = torch.tensor(             feature               ).to(dtype=torch.float32) 



#%% edge weighted

All_path = 'C://Users//yrt05//Desktop//Covid_coference_presentation//edge_all.csv'

DDI_mapping_add_PPI = {key_: val_+len(DDI_mapping)  for key_, val_ in PPI_mapping.items()} 

All_mapping = { **DDI_mapping, **DDI_mapping_add_PPI}  ### drug + prot


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

edge_index, edge_label = load_edge_csv(
    All_path,
    src_index_col='V1',
    src_mapping=All_mapping,
    dst_index_col='V2',
    dst_mapping=All_mapping,
    # encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

edge_index.size()   ### 51700 edges

# np.where(edge_index[0]<600)

edge_weight = np.exp(drug_prot_RWR[:,1].astype(float))  ### drug + prot
# i=0
data_edge_weight = np.zeros(len(edge_index[0]))
for i in range(len(edge_index[0])):
    ind_1 = int(edge_index[0][i])
    ind_2 = int(edge_index[1][i])
    prob = edge_weight[ind_1] * edge_weight[ind_2]
    data_edge_weight[i] = prob
    
data.edge_weight = data_edge_weight


metadata = data.metadata()


# edge_weight_prod = np.multiply(edge_weight[edge_index[0]], edge_weight[edge_index[1]])

# data.edge_weight = edge_weight_prod  ### drug + prot


train_data, val_data, test_data = transform(data)
data_label_index, _, _ = transform2(data)


from torch.nn import Parameter
from torch_geometric.nn import SGConv
from torch_geometric.nn import GAE, VGAE, GCNConv

    
#%% Feature

num_features = data_feature.shape[1]
# num_features = feature.shape[1]
in_channels = num_features
hidden_channels = 4
out_channels = 4

# model = VGAE(VariationalGCNEncoder(in_channels, hidden_channels, out_channels)).to(device)
# model = VGAE(VariationalGCNEncoder(data.num_nodes, in_channels = data_feature.shape[0], hidden_channels = 64, out_channels = 1)).to(device)
# model = VGAE(encoder=lambda x: (x, x))

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#%% Train def

threshold = 0.3
# drug_no_repeat = np.ones(100)
num_nodes = 963


       #%% Test   VGAE 
    class VariationalGCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
        # def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
            super().__init__()
            
            w = torch.empty(num_nodes, hidden_channels)
            random_seed = 1
            torch.manual_seed(random_seed)
            self.node_emb = Parameter(torch.nn.init.xavier_normal_(w))
            
            self.conv1 = GCNConv(in_channels, hidden_channels, cached = True)
            self.conv2 = GCNConv(hidden_channels, hidden_channels, cached = True)
    
            
            self.conv_mu = GCNConv(hidden_channels, out_channels, cached = True)  ### Transductive learning
            self.conv_logstd = GCNConv(hidden_channels, out_channels, cached = True)
    
            self.reset_parameters()
            # self.reparametrize()
            
        
        def reset_parameters(self):
            torch.nn.init.xavier_uniform_(self.node_emb)
            # torch.nn.init.xavier_uniform_(self.x)
            self.conv1.reset_parameters()
            self.conv2.reset_parameters() 
            self.conv_mu.reset_parameters() 
            self.conv_logstd.reset_parameters() 
        
        def forward(self, x, edge_index):
            x = self.node_emb
            x = self.conv1(x, edge_index).relu()
            # x = self.conv2(x, edge_index).relu()
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
    
    def train():
        model.train()
        optimizer.zero_grad()
        
        z = model.encode(train_data.x, train_data.edge_index)
        model.reparametrize(model.__mu__, model.__logstd__)
        
        loss = model.recon_loss(z, train_data.edge_index) # Negative sampling already in GAE model.recon_loss
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()    ### reconstruction loss & kl loss
        
    
        
        loss.backward()
        optimizer.step()
        return float(loss)
    
    
    model = VGAE(VariationalGCNEncoder(in_channels, hidden_channels, out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    @torch.no_grad()
    ### "data" is "train_data"
    def test(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
    
        out = model.decoder(z, data.edge_label_index).view(-1).sigmoid()   ### inner product and sigmoid
        
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy()), out
    
    
    val_auc_record = []
    test_auc_record = []
    loss_record = []
    
    
    epochs = 100
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
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
    print(f'{model}')
    
    ### Plot
    
    epo = np.linspace(1, epoch,epoch)
    # epo = np.linspace(1, 2*epoch+1, 2*epoch+1)
    plt.style.use('seaborn')
    plt.plot(epo, test_auc_record, 'r', label = 'test_auc_record')
    plt.plot(epo, torch.Tensor(loss_record)/  max(loss_record) , 'b', label = 'loss_record')
    plt.plot(epo, val_auc_record, 'g', label = 'val_auc_record')
    plt.xlabel('Epoch')
    plt.legend()
    
    
    ### make prediction
    z1 = model.encode(test_data.x, test_data.edge_index) # dim(Z) = [Node_num, out_channels] ,  each dim(z_i) = [1 , out_channels] 
    model.reparametrize(model.__mu__, model.__logstd__)
    
    
    prob_adj = z1 @ z1.t().sigmoid()
    
    # # hist = torch.histc(prob_adj[0,:])
    # hist = torch.histc(prob_adj)
    # print("Histogram of T:\n", hist)
    # # Visualize above calculated histogram 
    # # as bar diagram
    # bins = 5
    # x = range(bins)
    # plt.bar(x, hist, align='center', color=['forestgreen'])
    # plt.xlabel('Bins')
    # plt.ylabel('Frequency')
    # plt.show()
    
    
    
    max(prob_adj[0,:])
    
    
    
    final_edge_index = (prob_adj > threshold).nonzero(as_tuple=False).t()
    # final_edge_index = model.decode_all(z)
    
    # neg_edge = torch.empty(0,2)
    # model.test(z, test_data.edge_index, neg_edge)
    
    final_edge_index.size()
    
    edge_final_VGAE = final_edge_index.detach().numpy()
    
        
    #%% GCN : Link prediction:  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
    # in_channels = 32
    
if(True):
    
    params = {
    # 'layer': [1,2,3,4,5,6,7,8,9,10],
    'layer': [2,4,6,8,10,15,20,30],
    # 'lr': [0.001,0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'epochs': [200],
    'weight_dim':[10],
    }
    
    i_plot = 1
    
    # hidden_channels = 10
    out_channels = 10
    
    # import torch_geometric
    
    
    for layer in params['layer']:
        
        for hidden_channels in params['weight_dim']:
        
            for epochs in params['epochs']:
        
                class Net(torch.nn.Module):
                    def __init__(self,num_nodes, in_channels, hidden_channels, out_channels, num_layers, model_type='GCN'):
                        super().__init__()
                        w = torch.empty(num_nodes, hidden_channels)
                        random_seed = 1
                        torch.manual_seed(random_seed)
                        torch.cuda.manual_seed(random_seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False


                        self.node_emb = Parameter(torch.nn.init.xavier_normal_(w))
                        
                        conv_model = self.build_conv_model(model_type) ## GCNConv
                        self.convs = nn.ModuleList()
                        self.convs.append(conv_model(hidden_channels, hidden_channels, cached = True))
                        
                        # self.conv1 = GCNConv(hidden_channels, hidden_channels, cached = True)
                        # self.conv2 = GCNConv(hidden_channels, 2*hidden_channels, cached = True)
                        # self.conv3 = GCNConv(2*hidden_channels, out_channels, cached = True)
                        # self.convs = GCNConv(hidden_channels, hidden_channels, cached = True)
                        
                        for _ in range(num_layers - 1):
                            self.convs.append(conv_model(hidden_channels, hidden_channels, cached = True))
                        
                        self.reset_parameters()

                    ### For testing if parameter is reset for each layer
                    # A = nn.ModuleList()
                    # conv_model = pyg_nn.GCNConv ## GCNConv
                    # A.append(conv_model(hidden_channels, hidden_channels, cached = True))
                    # A.append(conv_model(hidden_channels, hidden_channels, cached = True))
                    # A.append(conv_model(hidden_channels, hidden_channels, cached = True))
                    # A.children()
                    # for layer in A.children():
                    #    if hasattr(layer, 'reset_parameters'):
                    #        print(layer)
                    #        layer.reset_parameters()           
                    # for layer in A.modules():
                    #     if hasattr(layer, 'reset_parameters'):
                    #         print(layer)
                    #         # layer.reset_parameters()  
                    # len(A.modules())

                    def reset_parameters(self):
                        torch.nn.init.xavier_uniform_(self.node_emb)
                        # self.convs.reset_parameters()
                        # self.apply(torch.nn.init.xavier_uniform_(self.node_emb))

                        for layer1 in self.children():
                           if hasattr(layer1, 'reset_parameters'):
                               layer1.reset_parameters()
                               
                               
                    def build_conv_model(self, model_type):
                        if model_type == 'GCN':
                            return pyg_nn.GCNConv
                        # elif model_type == 'GAT':
                        #     return pyg_nn.GATConv
                        # elif model_type == "GraphSage":
                        #     return pyg_nn.SAGEConv
                        else:
                            raise ValueError(f'Model {model_type} unavailable')
                            
                            
                
                    def encode(self, x, edge_index):  # forward
                        x = self.node_emb
                        
                        for i in range(len(self.convs)-1):
                            x = self.convs[i](x, edge_index).relu()
                            # x = F.dropout(x, p=0.1, training=self.training)
                            
                        x = self.convs[len(self.convs)-1](x, edge_index)
                        
                        cache = self._cached_adj_t
                        
                        return x
                    
                    
        
                
                    def decode(self, z, edge_label_index):
                        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
                
                
                
            
                num_nodes = 963
                num_layers = layer
                model = Net(num_nodes, num_features, hidden_channels, out_channels, num_layers).to(device)
                
                
                
                
                
                optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
                criterion = torch.nn.BCEWithLogitsLoss()
                
                
                def train():
                    model.train()
                    optimizer.zero_grad()
                    z = model.encode(train_data.x, train_data.edge_index)
                
                    # negative sampling for every training epoch:
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
                
                    out = model.decode(z, edge_label_index).view(-1).sigmoid()
                    loss = criterion(out, edge_label)
                    loss.backward()
                    optimizer.step()
                    return loss
                
                
                @torch.no_grad()
                def test(data1):
                    
                    # data1 = val_data
                    model.eval()
                    z = model.encode(data1.x, data1.edge_index)
                    
                    out = model.decode(z, data1.edge_label_index).view(-1).sigmoid()
                    val_loss = criterion(out, data1.edge_label)
                    return roc_auc_score(data1.edge_label.cpu().numpy(), out.cpu().numpy()), out, val_loss.item()
                
                val_auc_record = []
                test_auc_record = []
                loss_record = []
                val_loss_record = []
                
                
                best_val_auc = final_test_auc = 0
                for epoch in range(1, epochs):
                    loss = train()
                    val_auc,val_out, val_loss = test(val_data)
                    test_auc,test_out,_ = test(test_data)
                    
                    
                    test_auc_record.append(test_auc)
                    loss_record.append(loss)
                    val_auc_record.append(val_auc)
                    val_loss_record.append(val_loss)
                    
                    
                    
                    if val_auc > best_val_auc:
                        best_val = val_auc
                        final_test_auc = test_auc
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
                          f'Test: {test_auc:.4f}')
                
                print(f'Final Test: {final_test_auc:.4f}')
                
                
                
                
                plt.subplot(3,3, i_plot).set_title('Layers = ' + str(layer) + ', Epochs=' + str(epochs)  )
                i_plot = i_plot +1
                
                epo = np.linspace(1, epoch,epoch)
                plt.style.use('seaborn')
                plt.plot(epo, test_auc_record, 'r', label = 'test_auc')
                plt.plot(epo, val_auc_record, 'g', label = 'val_auc')
                
                # compare training and validation loss
                plt.plot(epo, val_loss_record, label = 'Validation Loss')
                # plt.plot(epo, torch.Tensor(loss_record)/  (max(loss_record).detach().numpy()) , 'b', label = 'Training Loss')
                plt.plot(epo, torch.Tensor(loss_record) , 'b', label = 'Training Loss')
                plt.xlabel('Epoch')
                plt.legend()
        
        
        
        
        
        
        
        
        
        
        # threshold = 0.3
        
        z2 = model.encode(data.x, data.edge_index)
        prob_adj = z2 @ z2.t().sigmoid()
        
        # prob_adj = model.decode(z2, data_label_index.edge_label_index).view(-1).sigmoid()
        # out = model.decode(z2, test_data.edge_label_index).view(-1).sigmoid()
        # test = z2[data_label_index.edge_label_index[0]] * z2[data_label_index.edge_label_index[1]]
        
        prob_distribution = np.array((prob_adj.flatten()).detach().numpy(), dtype='float')
        threshold = np.quantile(prob_distribution, 0.65)
        
        final_edge_index = (prob_adj > threshold).nonzero(as_tuple=False).t()
        # final_edge_index = (prob_adj).nonzero(as_tuple=False).t()

    
        final_edge_index.size()
        
        edge_final_GCN = final_edge_index.detach().numpy()
    
    
    
    
    
        #%% Find DTI edges
        
        edggggeee_1 = edge_final_GCN
        
        # edggggeee_2 = edge_final_VGAE
    
        # edggggeee = np.concatenate([edggggeee_1, edggggeee_2],1)
        
        edggggeee = edggggeee_1
        
        test = pd.DataFrame(edggggeee.T, columns= ['V1', 'V2'])
        
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
        
        
        df = pd.DataFrame(list(zip(listOfDrug, listOfProt)), columns =['parent_key', 'gene_name']) 
        
        df.shape
        
        df.drop_duplicates(subset=['parent_key']) ### Number of predicted drugs 
        
        df.drop_duplicates(subset=['gene_name']) ### Number of predicted prots 
        
        
        # df.to_excel (r'drug_prot.xlsx', index = False, header=True)
        
        
        
        df_VGAE = df
        
        df_GCN = df
        
        df_all = pd.concat([df_VGAE, df_GCN])
        
        # df_all[df_all.duplicated()]
        df_all[~df_all['parent_key'].duplicated()]
        
        drug_no_repeat = df_all[~df_all['parent_key'].duplicated()]
        # drug_no_repeat[~drug_no_repeat.drug == '']
        
        int_df = pd.merge(DTI, drug_no_repeat, how ='inner', on =['parent_key', 'gene_name'])
        
        
        print("Number of predicted Drugs", len(drug_no_repeat))
    
    
    
    
    
    
    # threshold = threshold + 0.1

len(drug_no_repeat)


#%% Resulting interactive net
import networkx as nx
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

G=nx.Graph(name='Protein Interaction Graph')
interactions = np.array(df_all)
for i in range(len(interactions)):
    interaction = interactions[i]
    a = interaction[0] # protein a node
    b = interaction[1] # protein b node
    w = 1 # score as weighted edge where high scores = low weight
    G.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph


# deg_centrality = nx.degree_centrality(G)
# degrees = [val for (node, val) in G.degree()]
# G_deg = nx.degree_histogram(G)


# %matplotlib qt
# pos = nx.spring_layout(G) # position the nodes using the spring layout
# # plt.figure(figsize=(11,11),facecolor=[0.7,0.7,0.7,0.4])
# plt.figure()
# nx.draw_networkx(G)
# plt.axis('off')
# plt.show()
dict(G.degree()).values()
dege = dict(G.degree())  ## Degree





import collections
degree_sequence=sorted([d for n,d in G.degree()], reverse=True) # degree sequence
#print "Degree sequence", degree_sequence
degreeCount=collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d+0.4 for d in deg])
ax.set_xticklabels(deg)



# draw graph in inset
plt.axes([0.4, 0.4, 0.5, 0.5])
Gcc=sorted((G.subgraph(c) for c in nx.connected_components(G)), key = len, reverse=True)[0]
pos=nx.spring_layout(G)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size=30)
nx.draw_networkx_edges(G, pos, alpha=0.5)

plt.savefig("degree_histogram.png")
plt.show()
