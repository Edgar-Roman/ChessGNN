# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dqJig6jJJtnizCFbRiiGbqvdn8qnFVyv
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive

drive.mount('/content/drive')

# cd into where the data file is stored
# %cd /content/drive/MyDrive/cs224w-project/

!pip install python-chess

import os

max_workers = os.cpu_count()

import chess

def calculate_material_balance(board):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    material_balance = 0
    for piece_type in piece_values:
        material_balance += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        material_balance -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return material_balance

def calculate_pawn_structure(board):
    # For simplicity, just counting the number of pawns for now.
    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    return white_pawns - black_pawns

def calculate_mobility(board):
    return board.legal_moves.count()

def get_squares_around(square):
    """ Get the squares around a given square. """
    surrounding_squares = []
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    for f in range(file - 1, file + 2):  # File of the square and its adjacent files
        for r in range(rank - 1, rank + 2):  # Rank of the square and its adjacent ranks
            if 0 <= f <= 7 and 0 <= r <= 7:  # Ensure the file and rank are within bounds
                adjacent_square = chess.square(f, r)
                if adjacent_square != square:
                    surrounding_squares.append(adjacent_square)

    return surrounding_squares

def evaluate_king_safety(board):
    """ Evaluate the safety of the king based on surrounding pawns. """
    king_safety = 0
    for color in [chess.WHITE, chess.BLACK]:
        king_square = board.king(color)
        pawn_squares = board.pieces(chess.PAWN, color)
        safety_count = sum(1 for sq in get_squares_around(king_square) if sq in pawn_squares)
        king_safety += safety_count if color == chess.WHITE else -safety_count
    return king_safety



def control_of_center(board):
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    center_control = 0
    for square in center_squares:
        if board.is_attacked_by(chess.WHITE, square):
            center_control += 1
        if board.is_attacked_by(chess.BLACK, square):
            center_control -= 1
    return center_control

def piece_development(board):
    """ Calculate the development of pieces from their starting positions. """
    development = 0

    # Initial positions for non-pawn pieces except the king.
    initial_positions = {
        chess.WHITE: [chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1],
        chess.BLACK: [chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8]
    }

    # Count developed pieces (those not in their initial positions).
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                if square not in initial_positions[color]:
                    development += 1 if color == chess.WHITE else -1

    return development


def extract_features(fen):
    board = chess.Board(fen)
    features = {
        "material_balance": calculate_material_balance(board),
        "pawn_structure": calculate_pawn_structure(board),
        "mobility": calculate_mobility(board),
        "king_safety": evaluate_king_safety(board),
        "center_control": control_of_center(board),
        "piece_development": piece_development(board)
    }
    return features

import chess
import pickle
import warnings
import concurrent.futures

#from extract_features import extract_features
import requests

from tqdm import tqdm


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
        response = requests.get(f'https://stockfish.online/api/stockfish.php?fen={fen}&depth={DEPTH}&mode={MODE}')
        if response.status_code == 200:
            #print("RESPONSEEEE", response.json())
            return response.json().get('data').split()[2]
        else:
            print(f"Error fetching data for FEN: {fen}. Status Code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request failed for FEN: {fen}. Error: {e}")
        return None

def process_game(game):
    all_game_labels = []
    for fen in game:
        x = create_feature_vector(fen)
        label = fetch_label(fen)
        if label is not None:
            all_game_labels.append((x, label))
    return all_game_labels

if __name__ == '__main__':
    file_path = '/content/drive/MyDrive/cs224w-project/subset.txt'
    chess_games_moves = preprocess_chess_data(file_path)
    subset_games_fen = games_to_fen(chess_games_moves)

    DEPTH = 5
    MODE = 'eval'

    all_games_labels = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_game, subset_games_fen), total=len(subset_games_fen)):
            all_games_labels.extend(result)

    x, y = zip(*all_games_labels)

    data = {'x': x, 'y': y}
    pickle.dump(data, open('/content/drive/MyDrive/cs224w-project/chess_db.zip/subset_games_test_updated.pkl', 'wb'))