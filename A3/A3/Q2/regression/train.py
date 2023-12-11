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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report ,roc_auc_score, roc_curve 
import numpy as np
import networkx as nx
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import to_networkx
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error

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
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(devic

        x_1 = self.gine_layer_1(x, edge_index, edge_attr)
        x_2 = self.gine_layer_2(x_1, edge_index, edge_attr)
        x_3 = self.gine_layer_3(x_2, edge_index, edge_attr)

        x_1 = global_add_pool(x_1, data.batch)
        x_2 = global_add_pool(x_2, data.batch)
        x_3 = global_add_pool(x_3, data.batch)

        x = self.mlp(torch.cat((x_1, x_2,x_3), dim=1))

        return x



class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        super(GraphDataset, self).__init__()

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        x = data.x.clone().detach().to(torch.float32)
        edge_index = data.edge_index.clone().detach().to(torch.long)
        y = data.y.clone().detach().to(torch.float32)

        num_nodes = data.num_nodes
        num_edges = data.num_edges

        if data.edge_attr is not None:
            edge_attr = data.edge_attr.clone().detach().to(torch.float32)
            return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes, num_edges=num_edges, edge_attr=edge_attr)
        else:
            return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes, num_edges=num_edges)


def train_GINE(dataset, num_epochs, lr, dropout_prob, batch_size, weight_decay=0.001):
    model = GraphClassifier(input_dim=9, hidden_dim=100, output_dim=1, dropout_prob=dropout_prob)
    model.to(device)

    train_data = [data.to(device) for data in dataset]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        scheduler.step()
    
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig('Q2_MSELoss.png')

    return model

def load_graph_data(source):
    node_features = pd.read_csv(f'{source}node_features.csv.gz', header=None).values.tolist()
    num_nodes = pd.read_csv( f'{source}num_nodes.csv.gz', header=None).values.tolist()
    edge_features = pd.read_csv( f'{source}edge_features.csv.gz', header=None).values.tolist()
    edges = pd.read_csv( f'{source}edges.csv.gz', header=None).values.tolist()
    graph_labels=pd.read_csv( f'{source}graph_labels.csv.gz', header=None).values.tolist()
    num_edges = pd.read_csv( f'{source}num_edges.csv.gz', header=None).values.tolist()
    valid_indices = ~np.isnan(graph_labels)
    node_features = np.array(node_features)
    edge_features = np.array(edge_features)


    node_features_normalized = PowerTransformer().fit_transform(node_features)

    edge_features_normalized = PowerTransformer().fit_transform(edge_features)

    data_list = []

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

            data_list.append(data)
        node_start+=graph_num_nodes
        edge_start+=graph_num_edges

    dataset = GraphDataset(data_list)
    return dataset

def train_regression(dataset):
    graph_features=[]
    graph_labels=[]
    for data in dataset:
         x, y, edge_attr = data.x.numpy(), data.y.numpy(), data.edge_attr.numpy()
         graph_node_features = np.mean(x, axis=0)
         if edge_attr is not None:
            graph_edge_features = np.mean(edge_attr, axis=0)
            graph_features.append(np.concatenate((graph_node_features, graph_edge_features)))
         else:
            graph_features.append(graph_node_features)
         graph_labels.append(y)
    graph_features = np.vstack(graph_features)
    graph_labels = np.concatenate(graph_labels)
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(graph_features, graph_labels)
    return linear_reg_model

def compare_with_baseline(gnn_model, linear_reg_model, val_dataset):
    gnn_model.eval()
    gnn_predictions = []
    linear_reg_predictions = []
    true_labels = []

    with torch.no_grad():
        for data in val_dataset:
            data = data.to(device)
            
            gnn_output = gnn_model(data)
            gnn_predictions.append(gnn_output.item())

            x, y, edge_attr = data.x.numpy(), data.y.numpy(), data.edge_attr.numpy()
            graph_node_features = np.mean(x, axis=0)
            if edge_attr is not None:
                graph_edge_features = np.mean(edge_attr, axis=0)
                graph_features = np.concatenate((graph_node_features, graph_edge_features))
            else:
                graph_features = graph_node_features

            linear_reg_output = linear_reg_model.predict(graph_features.reshape(1, -1))
            linear_reg_predictions.append(linear_reg_output)
            
            true_labels.append(data.y.item())

    gnn_predictions = np.array(gnn_predictions)
    linear_reg_predictions = np.array(linear_reg_predictions)
    true_labels = np.array(true_labels)

    # Plotting
    plt.plot(range(0,25), gnn_predictions[:25], label='GNN Predictions',linestyle='-', color='blue')
    plt.plot(range(0,25), linear_reg_predictions[:25], label='Linear Regression Predictions',linestyle='-', color='orange')
    plt.plot(range(0,25), true_labels[:25], label='Desired Labels', linestyle='-', color='green')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Comparison of Predictions')
    plt.legend()
    plt.savefig('Q2_baseline_regression.png')
    
    gnn_mse = mean_squared_error(true_labels, gnn_predictions)
    linear_reg_mse = mean_squared_error(true_labels, linear_reg_predictions)

    print(f'Mean Squared Error - GINE: {gnn_mse:.4f}, Linear Regression: {linear_reg_mse:.4f}')



def main():
    parser = argparse.ArgumentParser(description="Training a regression model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    args = parser.parse_args()
    print(f"Training a regression model. Output will be saved at. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}. Model will be saved at {args.model_path}")

    train_dataset=load_graph_data(args.dataset_path)
    val_dataset=load_graph_data(args.val_dataset_path)
    GINE_model = train_GINE(train_dataset, num_epochs=200, lr=0.001, dropout_prob=0.3, batch_size=100)
    lin_model = train_regression(train_dataset)
    compare_with_baseline(GINE_model,lin_model,val_dataset)

    torch.save(GINE_model, f'{args.model_path}GINE_regression_model.pth')

    print("Model Generated as \'GINE_regression_model.pth\'")

if __name__=="__main__":
     main()

