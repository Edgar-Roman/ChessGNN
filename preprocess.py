import chess
import pickle
import warnings

from extract_features import extract_features 


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

if __name__ == '__main__':
    file_path = 'data/subset.txt'
    chess_games_moves = preprocess_chess_data(file_path)
    subset_games_fen = games_to_fen(chess_games_moves)

    single_game_fen = subset_games_fen[0]

    import requests

    DEPTH = 5
    MODE = 'eval'

    for fen in single_game_fen:
        response = requests.get('https://stockfish.online/api/stockfish.php?fen={}&depth={}&mode={}'.format(fen, DEPTH, MODE))
        evaluation = response.json().get('data').split()[2]
        print(evaluation)

        print(extract_features(fen))




    # pickle.dump(subset_games_fen, open('data/subset_games_fen.pkl', 'wb'))