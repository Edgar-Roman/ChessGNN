import pickle
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from graph import create_data
from preprocess import process_chess_data

class GATChessModel(torch.nn.Module):
    def __init__(self, num_features, heads=8, dropout=0.6):
        super(GATChessModel, self).__init__()
        self.conv1 = GATConv(num_features, 32, heads=heads, dropout=dropout)
        self.conv2 = GATConv(32 * heads, 1, heads=1, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x


def get_edge_indices():
    file_path = 'data/subset.txt'
    _, exception_indices = process_chess_data(file_path)

    with open('data/subset_games_fen.pkl', 'rb') as f:
        node_id = 0
        fen_to_node = {}

        games = pickle.load(f)
        for i, game in enumerate(games):
            if i in exception_indices:  # skip games with exceptions
                continue
            for state in game:
                if state not in fen_to_node:
                    fen_to_node[state] = node_id
                    node_id += 1

        src, target = [], []
        for game in games:
            for i in range(len(game) - 1):
                src.append(fen_to_node[game[i]])
                target.append(fen_to_node[game[i+1]])
        
        edge_index = torch.tensor([src, target], dtype=torch.long)

        return edge_index


def get_data():
    with open('data/subset_games_test.pkl', 'rb') as f:
        data = pickle.load(f)
        flattened_list = [lst for sublist in data['x'] for lst in sublist]
        y = torch.tensor(data['y'], dtype=torch.float)
        edge_index = get_edge_indices()
        
        return Data(x=flattened_list, edge_index=edge_index, y=y)
    

if __name__ == '__main__':
    num_features = 12 * 64
    learning_rate = 0.01
    num_epochs = 100

    model = GATChessModel(num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()  # Mean Squared Error for regression
    
    data = get_data()


    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x)
        loss = loss_func(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

