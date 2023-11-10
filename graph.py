import torch
from torch_geometric.data import Data
import pickle

games_fen = pickle.load(open('data/subset_games_fen.pkl', 'rb'))

# A function to encode a FEN string into a numerical feature vector
def encode_fen(fen):
    # Initialize a binary vector for the board
    # 12 piece types (6 for white, 6 for black) x 64 squares
    board_vector = [0] * (12 * 64)
    
    # Define a mapping from piece symbols to their index offsets
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Fill the vector with the presence of pieces
    pieces = fen.split(' ')[0]  # Get the piece placement part of the FEN
    square = 0
    for char in pieces:
        if char.isdigit():  # Empty squares
            square += int(char)
        elif char in piece_to_index:  # Occupied square
            index = piece_to_index[char] * 64 + square
            board_vector[index] = 1
            square += 1
        elif char == '/':  # New rank
            continue
    
    return board_vector


# Encode all FENs and create a mapping from FEN to node index
fen_to_index = {}
node_features = []

for game_fens in games_fen:
    for fen in game_fens:
        if fen not in fen_to_index:
            fen_to_index[fen] = len(fen_to_index)
            encoded_fen = encode_fen(fen)
            node_features.append(encoded_fen)

# Convert node features to a tensor
node_features_tensor = torch.tensor(node_features, dtype=torch.float)

# Create edges based on the sequence of moves
edge_indices = []

for game_fens in games_fen:
    for i in range(len(game_fens) - 1):
        source = fen_to_index[game_fens[i]]
        target = fen_to_index[game_fens[i + 1]]
        edge_indices.append((source, target))

# Convert edge indices to a tensor
edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

# Create a PyG Data object
data = Data(x=node_features_tensor, edge_index=edge_index_tensor)


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Convert PyG Data object to NetworkX graph
G = to_networkx(data, to_undirected=True)

# Draw the graph using NetworkX
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
plt.title('Graph Representation of Chess Games')
plt.show()
