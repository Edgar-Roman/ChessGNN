import pickle
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch_scatter

from preprocess import process_chess_data
from extract_features import extract_features

import time
from tqdm import tqdm

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
    # _, exception_indices = process_chess_data(file_path)
    exception_indices = []

    # Read exception indices from file
    with open('data/exception_indices.txt', 'r') as file:
        for line in file:
            exception_indices.append(int(line.strip()))

    print("exceptions", exception_indices)

    with open('data/subset_games_test.pkl', 'rb') as f:
        node_id = 0
        fen_to_node = {}

        games = pickle.load(f)
        print("NUM GAME", len(games))
        for i, game in enumerate(games):
            if i in exception_indices:  # skip games with exceptions
                continue
            for state in game:
                if state not in fen_to_node:
                    fen_to_node[state] = node_id
                    node_id += 1
        
        print("NODE ID", node_id)

        src, target = [], []
        for j, game in enumerate(games):
            for i in range(len(game) - 1):
                if j in exception_indices:  # skip games with exceptions
                    continue
                else:
                    src.append(fen_to_node[game[i]])
                    target.append(fen_to_node[game[i+1]])
        
        edge_index = torch.tensor([src, target], dtype=torch.long)

        return edge_index


def get_data():
    with open('data/subset_games_test.pkl', 'rb') as f:
        data = pickle.load(f)
        # print(extract_features(data['fen'][0][0]))
        x = [extract_features(fen) for game in data['fen'] for fen in game]
        x = torch.tensor(torch.stack(x), dtype=torch.float32)
 
        y = []
        for i in data['y']:
            if type(i) != str:
                continue
            else:
                y.append(float(i))
        # y = [float(i) for i in data['y']]
        edge_index = torch.tensor(get_edge_indices(), dtype=torch.long)

        y = torch.tensor(y, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, y=y)
    

if __name__ == '__main__':
    num_features = 6
    learning_rate = 0.01
    num_epochs = 100

    model = GATChessModel(num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()  # Mean Squared Error for regression
    
    data = get_data()

    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        start_time = time.time()

        optimizer.zero_grad()
        out = model(data)
        loss = loss_func(out, data.y)
        loss.backward()
        optimizer.step()

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Duration: {epoch_duration:.2f} sec")

        # print(f"Epoch {epoch+1}, Loss: {loss.item()}")

