import pickle

# open data/subset_games_test.pkl
with open('data/subset_games_test.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data['x'][0])
    print(data['y'][0])

with open('data/subset_games_fen.pkl', 'rb') as f:
    node_id = 0
    fen_to_node = {}

    games = pickle.load(f)
    for game in games:
        for state in game:
            if state not in fen_to_node:
                fen_to_node[state] = node_id
                node_id += 1
    
    print(fen_to_node)
