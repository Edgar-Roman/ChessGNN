import chess
import pickle
import warnings
import concurrent.futures

from extract_features import extract_features
import requests


from tqdm import tqdm

session = requests.Session()

DEPTH = 5
MODE = 'eval'

def convert_moves_to_san(moves_str):
    # Split the string into individual moves, ignoring the white and black move indicators
    moves = moves_str.split()
    san_moves = []

    # Extract the standard algebraic notation (SAN) part of each move (ignoring the move number and color indicator)
    for move in moves:
        san_move = move.split('.')[1]  # Take only the move part after the dot
        san_moves.append(san_move)

    # Combine the moves into a single space-separated string
    san_moves_str = ' '.join(san_moves)
    return san_moves_str

def moves_to_fen(moves):
    board = chess.Board()
    fen_list = [board.fen()]  # Include the starting position

    for move_san in moves.split():
        move = board.parse_san(move_san)
        board.push(move)
        fen_list.append(board.fen())
    
    return fen_list

def preprocess_chess_data(file_path):
    # List to hold all games moves
    games_moves = []

    # Open the file and skip the first five lines of metadata
    with open(file_path, 'r') as file:
        for _ in range(5):
            next(file)  # Skip metadata lines

        # Process each game entry line
        for line in file:
            if '###' in line:
                # Split the line at '###' and take the second part, which contains the moves
                moves = line.split('###')[1].strip()
                # Store the moves in the list
                games_moves.append(moves)

    return games_moves

# Process the data to convert moves to FEN for each game
def games_to_fen(games_moves):
    all_games_fen = []
    for moves_str in games_moves:
        san_moves = convert_moves_to_san(moves_str)
        game_fen = moves_to_fen(san_moves)
        all_games_fen.append(game_fen)
    return all_games_fen


def create_feature_vector(fen):
    features = extract_features(fen)
    features_vector = [features["material_balance"], features["pawn_structure"], 
                       features["mobility"], features["king_safety"], 
                       features["center_control"], features["piece_development"]]
    
    return features_vector


def fetch_label(fen):
    """
    Fetch the label for a given FEN from the API.
    """
    try:
        url = f'https://stockfish.online/api/stockfish.php?fen={fen}&depth={DEPTH}&mode={MODE}'
        response = session.get(url)
        if response.status_code == 200:
            return response.json().get('data').split()[2], False
        else:
            print(f"Error fetching data for FEN: {fen}. Status Code: {response.status_code}")
            return None, True
    except requests.RequestException as e:
        print(f"Request failed for FEN: {fen}. Error: {e}")
        return None, True
    

import concurrent.futures

NUM_THREADS = 5

def fetch_label_with_index(index, fen):
    try:
        label, exception_occurred = fetch_label(fen)
        return index, label, exception_occurred
    except Exception as e:
        return index, None, True

def process_chess_data(file_path):
    chess_games_moves = preprocess_chess_data(file_path)
    subset_games_fen = games_to_fen(chess_games_moves)

    all_labels = [None] * sum(len(game) for game in subset_games_fen)  # Pre-allocate space
    exception_indices = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []

        for game_index, game in enumerate(subset_games_fen):
            for i, fen in enumerate(game):
                fen_index = sum(len(subset_games_fen[j]) for j in range(game_index)) + i
                futures.append(executor.submit(fetch_label_with_index, fen_index, fen))

        # Wrap concurrent.futures.as_completed with tqdm for progress tracking
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            fen_index, label, exception_occurred = future.result()
            all_labels[fen_index] = label

            if exception_occurred:
                exception_indices.append(fen_index)

    # Reorganize all_labels to match the structure of subset_games_fen
    organized_labels = []
    start = 0
    for game in subset_games_fen:
        end = start + len(game)
        organized_labels.append(all_labels[start:end])
        start = end

    data = {'fen': subset_games_fen, 'y': organized_labels}
    return data, exception_indices


    # Reorganize all_labels to match the structure of subset_games_fen
    organized_labels = []
    start = 0
    for game in subset_games_fen:
        end = start + len(game)
        organized_labels.append(all_labels[start:end])
        start = end

    data = {'fen': subset_games_fen, 'y': organized_labels}
    return data, exception_indices

# Rest of the main script remains unchanged



if __name__ == '__main__':
    file_path = 'data/subset.txt'
    data, exception_indices = process_chess_data(file_path)
    pickle.dump(data, open('data/subset_games_test.pkl', 'wb'))