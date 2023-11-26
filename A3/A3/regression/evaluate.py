import argparse
import os
import gzip
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,PowerTransformer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, GINEConv
from torch_geometric.nn import ChebConv
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super(GraphClassifier, self).__init__()

        self.gine_layer_1 = GINEConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

        ), train_eps=False ,edge_dim=3)

        self.gine_layer_2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

         #   nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

        ), train_eps=False ,edge_dim=3)

        self.gine_layer_3 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

        ), train_eps=False ,edge_dim=3)

        self.mlp = nn.Sequential(
            nn.Linear(3*hidden_dim, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, output_dim)

        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)


        x_1 = self.gine_layer_1(x, edge_index, edge_attr)
        x_2 = self.gine_layer_2(x_1, edge_index, edge_attr)
        x_3 = self.gine_layer_3(x_2, edge_index, edge_attr)

        x_1 = global_add_pool(x_1, data.batch)
        x_2 = global_add_pool(x_2, data.batch)
        x_3 = global_add_pool(x_3, data.batch)

        x = self.mlp(torch.cat((x_1, x_2,x_3), dim=1))

        return x

def test(model, test_loader):
    model.eval()
    all_ys = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outpt = model(batch)
            output = outpt[0][0]
            all_ys.append(output)
    numpy_ys = np.asarray(all_ys)
    tocsv(numpy_ys, task="regression")

def tocsv(y_arr, *, task):
    assert task in ["classification", "regression"], f"task must be either \"classification\" or \"regression\". Found: {task}"
    assert isinstance(y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze().shape) == 1, f"y_arr must be a vector. shape found: {y_arr.shape}"
    assert not os.path.isfile(f"y_{task}.csv"), f"File already exists. Ensure you are not calling this function multiple times (e.g. when looping over batches). Read the docstring. Found: y_{task}.csv"
    y_arr = y_arr.squeeze()
    df = pd.DataFrame(y_arr)
    df.to_csv(f"y_{task}.csv", index=False, header=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluating the regression model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    print(f"Evaluating the regression model.Test dataset will be loaded from {args.dataset_path} , model will be loaded from {args.model_path}")
    GINE_model = torch.load(f'{args.model_path}GINE_regression_model.pth')
    node_features = pd.read_csv(f'{args.dataset_path}node_features.csv.gz', header=None).values.tolist()
    num_nodes = pd.read_csv( f'{args.dataset_path}num_nodes.csv.gz', header=None).values.tolist()
    edge_features = pd.read_csv( f'{args.dataset_path}edge_features.csv.gz', header=None).values.tolist()
    edges = pd.read_csv( f'{args.dataset_path}edges.csv.gz', header=None).values.tolist()
    graph_labels=pd.read_csv( f'{args.dataset_path}graph_labels.csv.gz', header=None).values.tolist()
    num_edges = pd.read_csv( f'{args.dataset_path}num_edges.csv.gz', header=None).values.tolist()
    valid_indices = ~np.isnan(graph_labels)
    node_features = np.array(node_features)
    edge_features = np.array(edge_features)

    node_features_normalized = PowerTransformer().fit_transform(node_features)

    edge_features_normalized = PowerTransformer().fit_transform(edge_features)

    train_data_list = []

    node_start=0
    edge_start=0
    for i in range(len(num_nodes)):
        graph_num_nodes = num_nodes[i][0]
        graph_num_edges = num_edges[i][0]

        if valid_indices[i]:
          graph_node_features=[]
          graph_edge_features=[]
          graph_edge_index=[]
          for j in range(node_start,node_start+graph_num_nodes):
              graph_node_features.append(node_features_normalized[j])
          for k in range(edge_start,edge_start+graph_num_edges):
              graph_edge_index.append(edges[k])
              graph_edge_features.append(edge_features_normalized[k])
          graph_graph_label = graph_labels[i]



          x = torch.tensor(graph_node_features, dtype=torch.float32)
          y = torch.tensor(graph_graph_label, dtype=torch.float32)
          edge_index = torch.tensor(graph_edge_index, dtype=torch.long).t().contiguous()


          data = Data(x=x, edge_index=edge_index, y=y, num_nodes=graph_num_nodes, num_edges=graph_num_edges)


          if len(graph_edge_features)>0:
              edge_attr = torch.tensor(graph_edge_features, dtype=torch.float32)
              data.edge_attr = edge_attr
          else:
              dummy_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
              data.edge_index = dummy_edge_index
              data.edge_attr = torch.tensor(np.asarray([[0,0,0]]), dtype=torch.float32)
              data.num_edges+=1

          train_data_list.append(data)
        node_start+=graph_num_nodes
        edge_start+=graph_num_edges
    
    test_loader = DataLoader(train_data_list, batch_size=1, shuffle=False)
    test(GINE_model,test_loader)

if __name__=="__main__":
    main()
