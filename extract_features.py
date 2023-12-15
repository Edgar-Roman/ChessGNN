import chess
import torch

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

    features_vector = [features["material_balance"], features["pawn_structure"], 
                    features["mobility"], features["king_safety"], 
                    features["center_control"], features["piece_development"]]
    
    return torch.tensor(features_vector)