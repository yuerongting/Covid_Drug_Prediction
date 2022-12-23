# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 03:53:50 2022

@author: yrt05
"""

# # import 
# # data.x_dict['Drug']

# # x_dict = data.x_dict
# # edge_index_dict = data.edge_index_dict


# # metadata = data.metadata()
# # for edge_type in metadata[1]:
# #     print(edge_type)

# # convs = torch.nn.ModuleList()
# # conv = HeteroConv({
# #                 # ('Prot', 'interact', 'Prot'): GCNConv(-1, hidden_channels, add_self_loops=False),
# #                 # ('Drug', 'interact', 'Drug'): GCNConv(-1, hidden_channels, add_self_loops=False),
# #                 ('Drug', 'bind', 'Prot'): GraphConv(-1, hidden_channels)#, add_self_loops=False),
# #                 # ('Drug', 'bind', 'Prot'): SAGEConv((-1, -1), hidden_channels),
# #             }, aggr='sum')
# # convs.append(conv)
# # conv(x_dict, edge_index_dict)
# conv(data.x, edge_index)

# num_layers = 1
# encoder = torch.nn.ModuleList()
# for _ in range(num_layers):
#     conv = HeteroConv({
#         edge_type: GraphConv(-1, hidden_channels)
#         for edge_type in metadata[1]
#     })
#     encoder.append(conv)
# lin = Linear(hidden_channels, out_channels)

# lin(x_dict['Drug'])



# for conv in encoder:
#     print(conv)
#     # x_test = conv(x, edge_index) #.relu()   
#     x_test = conv(x_dict, edge_index_dict)
#     x = conv(x, edge_index)

import pickle
with open("C:\\Users\\yrt05\\Desktop\\Covid_coference_presentation\\data_11_16 - Copy\\data_11_16.pickle", 'rb') as f:
    data_temp = pickle.load(f)
data_temp.values()
data_temp.keys()

data = data_temp['data']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

metadata = data.metadata()
train_data = data_temp['train_data']
val_data = data_temp['val_data']
test_data = data_temp['test_data']
                    
                    
                    
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        w = torch.empty(963, hidden_channels)
        random_seed = 1
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
        self.node_emb = Parameter(torch.nn.init.xavier_normal_(w))
            
                
        self.encoder = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GraphConv(-1, hidden_channels, cached = True)
                for edge_type in metadata[1]
            })
            self.encoder.append(conv)
        # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')
        # self.lin = Linear(hidden_channels, out_channels)
        
        self.reset_parameters()
        
                
    
        # ## For testing if parameter is reset for each layer
        # A = nn.ModuleList()
        # conv_model = pyg_nn.GCNConv ## GCNConv
        # A.append(conv_model(hidden_channels, hidden_channels, cached = True))
        # A.append(conv_model(hidden_channels, hidden_channels, cached = True))
        # # A.append(conv_model(hidden_channels, hidden_channels, cached = True))
        # A.children()
        # for layer in A.children():
        #     if hasattr(layer, 'reset_parameters'):
        #         print(layer)
        #         layer.reset_parameters()           
        # for layer in A.modules():
        #     if hasattr(layer, 'reset_parameters'):
        #         print(layer)
        #         # layer.reset_parameters()  
        # len(A.modules())
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        for layer1 in self.children():
           if hasattr(layer1, 'reset_parameters'):
               layer1.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        for conv in self.encoder:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            # x_dict = self.lin(x_dict['Drug'])

        return x_dict


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['Drug'][row], z_dict['Prot'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)    



class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, num_layers)
        # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

loss_def = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                  train_data['Drug', 'Prot'].edge_label_index)
    pred = pred.clamp(min=0, max=1)
    
    target = train_data['Drug', 'Prot'].edge_label.int()
    # loss = weighted_mse_loss(pred, target.int(), weight)
    loss = loss_def(pred, target.float())
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data_in):
    model.eval()
    pred = model(data_in.x_dict, data_in.edge_index_dict,
                  data_in['Drug', 'Prot'].edge_label_index)
    pred = pred.clamp(min=0, max=1)
    target = data_in['Drug', 'Prot'].edge_label.int()
    # rmse = F.mse_loss(pred, target).sqrt()
    rmse = loss_def(pred, target.float())
    return float(rmse)

#%% Grid search

  
params = {
    # 'layer': [1, 2, 4, 8],
    # 'epochs': [50],
    # 'learning_rate':[0.01],
    # 'hidden_channels' :[2,4,8,16]
    
    # 'layer': [2, 3, 4, 5],
    # 'epochs': [50],
    # 'learning_rate':[0.001],
    # 'hidden_channels' :[ 16, 32, 64, 128 ],
    
    
    'layer': [2, 3],
    'epochs': [50],
    'hidden_channels':[16, 32],
    'learning_rate':[0.001],
}

# i_plot = 1

best_val_record = []
layer_record = []
learnng_rate_record = []

# hidden_channels = 4
# out_channels = 4

import torch_geometric





#%% Hyper-parameter tuning
training_record = []
valid_record = []
test_record = []
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
                
                model = Model(hidden_channels=hidden_channels, num_layers = layer).to(device)
                
                with torch.no_grad():  # Initialize lazy modules.
                     model.encoder(train_data.x_dict, train_data.edge_index_dict)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                for epoch in range(1, epochs):
                    loss = train()
                    train_rmse = test(train_data) # actually "binary cross entropy", not "rmse"
                    val_rmse = test(val_data)
                    test_rmse = test(test_data)
                    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
                    #       f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
                    
                    
                    # Loss recording
                    training_record.append(train_rmse)
                    valid_record.append(val_rmse)
                    test_record.append(test_rmse)
                    
                    
                    if epoch <= 1:
                        lowest_training_loss = train_rmse
                        lowest_test_loss = test_rmse
                    else:
                        if lowest_training_loss >= train_rmse:
                            lowest_training_loss = train_rmse
                        if lowest_test_loss >= test_rmse:
                            lowest_test_loss = test_rmse
                
                loss_train_record.append(lowest_training_loss) # data1
                loss_test_record.append(lowest_test_loss) # data2
                hyper_hidden_channel.append(hidden_channels) # axis
                hyper_layer.append(layer) # axis
                





#%% Grid search plot
x = hyper_hidden_channel
y = hyper_layer       
z = loss_test_record  # z[hyper_layer][hyper_hidden_channel]
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, len(z)))
fig = plt.figure()
# ax = fig.add_subplot(projection='2d')
ax = fig.add_subplot()
scats = ax.scatter(x, y, s=30, c = z, marker = 'o', cmap = "hsv")
fig.colorbar(scats)
plt.xlabel('Number of Hidden Channels', fontsize = 16)
plt.ylabel('Number of Layers', fontsize = 16)
plt.show()

import numpy as np
index_min = np.argmin(z) 
print("Min loss index", index_min)
print("Min loss", loss_test_record[index_min])
print("Layer number", y[index_min])
print("Channel number", x[index_min])




#%% Optimal Net


model = Model(hidden_channels=x[index_min], num_layers = y[index_min]).to(device)
                
with torch.no_grad():  # Initialize lazy modules.
     model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

training_record = []
valid_record = []
test_record = []

for epoch in range(1, 300):
    loss = train()
    train_rmse = test(train_data) # actually "binary cross entropy", not "rmse"
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
    #       f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
    
    
    # Loss recording
    training_record.append(train_rmse)
    valid_record.append(val_rmse)
    test_record.append(test_rmse)
    # if test_rmse == min_test_record:
    #     min_test_epoch = epochs
    #     print(epochs)
    
# min_test_record = min(test_record)
min_test_idx = np.argmin(test_record)
print(min_test_idx)


plt.plot(training_record, label = 'Training Loss')
plt.plot(valid_record, label = 'Validation Loss')
plt.legend(fontsize = 16)

plt.xlabel('Epochs', fontsize = 16)
# plt.ylabel('Number of Layers', fontsize = 16)
plt.show()


# Stop at min_test_epoch
model = Model(hidden_channels=x[index_min], num_layers = y[index_min]).to(device)
                
with torch.no_grad():  # Initialize lazy modules.
     model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

training_record = []
valid_record = []


for epoch in range(1, min_test_idx+1):
    loss = train()
    train_rmse = test(train_data) # actually "binary cross entropy", not "rmse"
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
    #       f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
    
    
    # Loss recording
    training_record.append(train_rmse)
    valid_record.append(val_rmse)
    test_record.append(test_rmse)
    # if test_rmse == min_test_record:
    #     min_test_epoch = epochs
    #     print(epochs)




#
from tqdm import tqdm
total_Drugs = 468
total_Prots = 495

percent_pred = 0.99
DTI_recs = []
DTI_record = torch.tensor([])
num_most_prossible = 5

for prot_id in tqdm(range(0, total_Prots)):
    prot_ids = torch.tensor([prot_id] * total_Drugs)
    drug_ids = torch.arange(total_Drugs)
    edge_label_index = torch.stack([drug_ids, prot_ids], dim=0)
    pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
    
    torch.histogram(pred)
    quantile = torch.quantile(pred.data, percent_pred, keepdim=True)
    # bins = 5
    # hist = torch.histc(pred, bins = bins, min = 0, max = 1)

    # # Visualize above calculated histogram as bar diagram
    
    # x = range(bins)
    # plt.bar(x, hist, align='center')
    # plt.xlabel('Bins')
    # plt.ylabel('Frequency')
    # plt.show()
    
    #% Choose based on quantile
    pred = pred.clamp(min=0, max=1)
    rec_movie_ids = (pred > percent_pred).nonzero(as_tuple=True)[0].tolist()[:num_most_prossible]
    top_ten_recs = [rec_movies for rec_movies in rec_movie_ids] 

    
    #% Predict edges
    prot_pred = torch.tensor([prot_id] * len(rec_movie_ids) )
    drug_pred = torch.tensor(rec_movie_ids)
    DTI_edge_pred = torch.stack([drug_pred, prot_pred], dim=0)
    # prot_record
    
    DTI_record = torch.cat((DTI_record, DTI_edge_pred), 1)
    DTI_record = DTI_record.int()   # drug & prot
    
    DTI_recs.append({'Prot_id': prot_id, 'Top Drugs': top_ten_recs})
    
    
DDI_mapping = data_temp['DDI_mapping']
PPI_mapping = data_temp['PPI_mapping']

    
DTI_no_empty = []
for i in range(len(DTI_recs)):
    
    if len(DTI_recs[i]['Top Drugs'])!=0:
        DTI_no_empty.append(DTI_recs[i])
        
DTI_no_empty = pd.DataFrame(DTI_no_empty)



DTI_values = DTI_no_empty.values

# DTI_values[:,0]
# DTI_values[:,1][0][:]

DTI_list = []
for i in range( len(DTI_values[:,1]) ):
    for j in range( len(DTI_values[:,1][i])  ):
        # for m in range(len(DTI_values[:,1][i][j])):
            DTI_list.append(DTI_values[:,1][i][j])


### Count frequency            
import collections
threshold_freq_percent = 0.15  ### percent
threshold_freq = round(495 * threshold_freq_percent)
a = DTI_list
counter = collections.Counter(a)
print(len(counter))

# counter.keys()
d = counter
d = dict((k, v) for k, v in d.items() if v >= threshold_freq)


size = 18
plt.bar(counter.keys(), counter.values())
plt.scatter(d.keys(), d.values())
plt.xlabel('Drug Index', fontsize = size)
plt.ylabel('Drug Frequency', fontsize = size)
plt.xticks(fontsize = size)
plt.yticks(fontsize = size)
# plt.xlim(0,270)
plt.plot(counter.keys(),threshold_freq*np.ones(len(counter.keys())), 'r')
# plt.annotate('Mimosine', xy = (110, 60), 
#              fontsize = 20, xytext = (90, 80), 
#              arrowprops = dict(facecolor = 'red'),
#              color = 'b')
plt.annotate('Threshold: 15%', xy = (110, threshold_freq), 
             fontsize = 20, xytext = (130, 20+threshold_freq), 
             arrowprops = dict(facecolor = 'red'),
             color = 'b')
plt.grid()
plt.show()  

DTI_list = list(d.keys())

### Screen            
DTI_list = [x for n, x in enumerate(DTI_list) if x not in DTI_list[:n]]
            
final_drugs = []
for i in DTI_list:
    final_drugs.append(list(DDI_mapping.keys())[i])


print(final_drugs)



# for i in final_drugs:
#     print(i)

drug_test = DDI_mapping['DB00180']
# drug_test = DDI_mapping['DB00210']


# DTI_list = []
myset = set()
for i in range( len(DTI_values[:,1]) ):
    if drug_test in DTI_values[:,1][i]:
        myset.add(DTI_values[:,0][i])
        
        
final_prot = []
for i in myset:
    final_prot.append(list(PPI_mapping.keys())[i])


with open('final_prot.txt', 'w') as file: 
    file.writelines("% s\n" % data for data in final_prot)
# final_drugs = DDI_mapping[DTI_list[:]]


import networkx as nx
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

protein_list = final_prot
proteins = '%0d'.join(protein_list)
url = 'https://string-db.org/api/tsv/network?identifiers=' + proteins + '&species=9606'
r = requests.get(url)



lines = r.text.split('\n') # pull the text from the response object and split based on new lines
data = [l.split('\t') for l in lines] # split each line into its components based on tabs
# convert to dataframe using the first row as the column names; drop empty, final row
df = pd.DataFrame(data[1:-1], columns = data[0]) 
# dataframe with the preferred names of the two proteins and the score of the interaction
interactions = df[['preferredName_A', 'preferredName_B', 'score']]  

G=nx.Graph(name='Protein Interaction Graph')
interactions = np.array(interactions)
for i in range(len(interactions)):
    interaction = interactions[i]
    a = interaction[0] # protein a node
    b = interaction[1] # protein b node
    w = float(interaction[2]) # score as weighted edge where high scores = low weight
    G.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph

# pos = nx.spring_layout(G) # position the nodes using the spring layout
# plt.figure(figsize=(11,11),facecolor=[0.7,0.7,0.7,0.4])
# nx.draw_networkx(G)
# plt.axis('off')
# plt.show()


def rescale(l,newmin,newmax):
    arr = list(l)
    return [(x-min(arr))/(max(arr)-min(arr))*(newmax-newmin)+newmin for x in arr]
# use the matplotlib plasma colormap
graph_colormap = cm.get_cmap('plasma', 12)
# node color varies with Degree
c = rescale([G.degree(v) for v in G],0.0,0.9) 
c = [graph_colormap(i) for i in c]
# node size varies with betweeness centrality - map to range [10,100] 
bc = nx.betweenness_centrality(G) # betweeness centrality
s =  rescale([v for v in bc.values()],1500,7000)
# edge width shows 1-weight to convert cost back to strength of interaction 
ew = rescale([float(G[u][v]['weight']) for u,v in G.edges],0.1,4)
# edge color also shows weight
ec = rescale([float(G[u][v]['weight']) for u,v in G.edges],0.1,1)
ec = [graph_colormap(i) for i in ec]


# %matplotlib qt  

import math
s = [i * 0.2 for i in s]
pos = nx.spring_layout(G, k=6/math.sqrt(G.order()))
# plt.figure(facecolor=[0, 0, 0, 0])
nx.draw_networkx(G, pos=pos, with_labels=False, node_color=c, node_size=s, edge_color= ec,width=ew,
                  font_color='white',font_weight='bold',font_size='7')
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax = plt.axes()
 
# Setting the background color of the plot
# using set_facecolor() method
# ax.set_facecolor("white")

hubs = nx.degree_centrality(G)
### Second hub
# unique_integers = hubs
# largest_integer = max(hubs.values()) 
# unique_integers = dict((k, v) for k, v in unique_integers.items() if v != largest_integer)
# second_largest_integer = max(unique_integers.values()) 
# hubs_2 = dict((k, v) for k, v in hubs.items() if v == second_largest_integer)
# max_two_hub = 


# nx.draw_networkx(Node, pos=nx.spring_layout(Node, 25), bg_color=[1,1,1,1], alpha=0.1, with_labels=False, node_size=100, node_color="green")
# ax.set_facecolor("blue")
hubs = dict((k, v) for k, v in hubs.items() if v == max(hubs.values()))
labels = {}    
for node in G.nodes():
    if node in hubs:
        #set the node name as the key and the label as its value 
        labels[node] = node
nx.draw_networkx_labels(G,pos,labels,font_size=30,font_color='blue')
plt.axis('off')
# ax = plt.axes()
# ax.set_facecolor("yellow")
plt.show()

print(hubs)




G.degree
sorted(G.degree, key=lambda x: x[1], reverse=True)


drug1_prot = G.nodes


# plt.savefig('PPI_select_DB00180.png', transparent=True, bbox_inches='tight')
# plt.savefig('PPI_select_DB00210.png', transparent=True, bbox_inches='tight')
  



        