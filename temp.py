import pickle

# open data/subset_games_test.pkl
with open('data/subset_games_test.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data['x'][0])
    print(data['y'][0])