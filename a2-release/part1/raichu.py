#
# raichu.py : Play the game of Raichu
#
# dfranci prebello bhanaraya [Dilip Nikhil Francies, Prinston Rebello, Bhanuprakash N]

import sys
import time
import copy

EMPTY = '.'
WHITE_PICHU = 'w'
WHITE_PIKACHU = 'W'
WHITE_RAICHU = '@'
BLACK_PICHU = 'b'
BLACK_PIKACHU = 'B'
BLACK_RAICHU = '$'

def board_to_string(board, N):
    return "\n".join(board[i:i+N] for i in range(0, len(board), N))

def create_solvable_board(board, N):
    tracker = 0
    transformed_board = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            transformed_board[i][j] = board[tracker]
            tracker += 1
    return transformed_board

def get_valid_pichu_moves(board, piece, row, col, N):
    moves = []
    if piece == WHITE_PICHU:
        directions = [(1,1),(1,-1)]
        for dirx, diry in directions:
            temp_board = copy.deepcopy(board)
            next_row = row + dirx
            next_col = col + diry
            if 0 <= next_row < N and 0 <= next_col < N:
                if temp_board[next_row][next_col] == EMPTY:
                    if next_row == N-1:
                        temp_board[row][col] = EMPTY
                        temp_board[next_row][next_col] = WHITE_RAICHU
                        moves.append(temp_board)
                    else:
                        temp_board[row][col] = EMPTY
                        temp_board[next_row][next_col] = piece
                        moves.append(temp_board)
                elif temp_board[next_row][next_col] == BLACK_PICHU: 
                    jump_row = row + (2 * dirx)
                    jump_col = col + (2 * diry)
                    if 0 <= jump_row < N and 0 <= jump_col < N and temp_board[jump_row][jump_col] == EMPTY:
                        temp_board[row][col] = EMPTY
                        temp_board[next_row][next_col] = EMPTY
                        if jump_row == N-1:
                            temp_board[jump_row][jump_col] = WHITE_RAICHU
                        else: 
                            temp_board[jump_row][jump_col] = piece
                        moves.append(temp_board)

    elif piece == BLACK_PICHU:
        directions = [(-1,1),(-1,-1)]
        for dirx, diry in directions:
            temp_board = copy.deepcopy(board)
            next_row = row + dirx
            next_col = col + diry
            if 0 <= next_row < N and 0 <= next_col < N:
                if temp_board[next_row][next_col] == EMPTY:
                    if next_row == 0:
                        temp_board[row][col] = EMPTY
                        temp_board[next_row][next_col] = BLACK_RAICHU
                        moves.append(temp_board)
                    else:
                        temp_board[row][col] = EMPTY
                        temp_board[next_row][next_col] = piece
                        moves.append(temp_board)
                elif temp_board[next_row][next_col] == WHITE_PICHU: 
                    jump_row = row + (2 * dirx)
                    jump_col = col + (2 * diry)
                    if 0 <= jump_row < N and 0 <= jump_col < N and temp_board[jump_row][jump_col] == EMPTY:
                        temp_board[row][col] = EMPTY
                        temp_board[next_row][next_col] = EMPTY
                        if jump_row == 0:
                             temp_board[jump_row][jump_col] = BLACK_RAICHU
                        else:
                            temp_board[jump_row][jump_col] = piece
                        moves.append(temp_board)
    return moves

def get_valid_pikachu_moves(board, piece, row, col, N):
    moves = []
    if piece == WHITE_PIKACHU:
        directions = [(1,0),(0,1),(0,-1)]
        for dirx, diry in directions:
            for distance in range(1,3):
                temp_board = copy.deepcopy(board)
                next_row = row + distance * dirx
                next_col = col + distance * diry
                if 0 <= next_row < N and 0 <= next_col < N:
                    if temp_board[next_row][next_col] == EMPTY:
                        if distance > 1:
                            check_empty_bw = all(temp_board[row + step * dirx][col + step * diry] is EMPTY for step in range(1,distance))
                            if check_empty_bw:
                                temp_board[row][col] = EMPTY
                                if next_row == N-1:
                                    temp_board[next_row][next_col] = WHITE_RAICHU
                                else:
                                    temp_board[next_row][next_col] = piece
                                moves.append(temp_board)
                        else:
                            temp_board[row][col] = EMPTY
                            if next_row == N-1:
                                temp_board[next_row][next_col] = WHITE_RAICHU
                            else:
                                temp_board[next_row][next_col] = piece
                            moves.append(temp_board)
                    elif temp_board[next_row][next_col] in (BLACK_PICHU, BLACK_PIKACHU):
                        jump_row = row + (distance + 1) * dirx
                        jump_col = col + (distance + 1) * diry
                        if 0 <= jump_row < N and 0 <= jump_col < N and temp_board[jump_row][jump_col] == EMPTY:
                            if distance > 1:
                                check_empty_bw = all(temp_board[row + step * dirx][col + step * diry] is EMPTY for step in range(1,distance))
                                if check_empty_bw:
                                    temp_board[row][col] = EMPTY
                                    temp_board[next_row][next_col] = EMPTY
                                    if jump_row == N-1:
                                        temp_board[jump_row][jump_col] = WHITE_RAICHU
                                    else:
                                        temp_board[jump_row][jump_col] = piece
                                    moves.append(temp_board)
                            else:
                                temp_board[row][col] = EMPTY
                                temp_board[next_row][next_col] = EMPTY
                                if jump_row == N-1:
                                    temp_board[jump_row][jump_col] = WHITE_RAICHU
                                else:
                                    temp_board[jump_row][jump_col] = piece
                                moves.append(temp_board)

    elif piece == BLACK_PIKACHU:
        directions = [(-1,0),(0,1),(0,-1)]
        for dirx, diry in directions:
            for distance in range(1,3):
                temp_board = copy.deepcopy(board)
                next_row = row + distance * dirx
                next_col = col + distance * diry
                if 0 <= next_row < N and 0 <= next_col < N:
                    if temp_board[next_row][next_col] == EMPTY:
                        if distance > 1:
                            check_empty_bw = all(temp_board[row + step * dirx][col + step * diry] is EMPTY for step in range(1,distance))
                            if check_empty_bw:
                                temp_board[row][col] = EMPTY
                                if next_row == 0:
                                    temp_board[next_row][next_col] = BLACK_RAICHU
                                else:
                                    temp_board[next_row][next_col] = piece
                                moves.append(temp_board)
                        else:
                            temp_board[row][col] = EMPTY
                            if next_row == 0:
                                temp_board[next_row][next_col] = BLACK_RAICHU
                            else:
                                temp_board[next_row][next_col] = piece
                            moves.append(temp_board)
                    elif temp_board[next_row][next_col] in (WHITE_PICHU, WHITE_PIKACHU):
                        jump_row = row + (distance + 1) * dirx
                        jump_col = col + (distance + 1) * diry
                        if 0 <= jump_row < N and 0 <= jump_col < N and temp_board[jump_row][jump_col] == EMPTY:
                            if distance > 1:
                                check_empty_bw = all(temp_board[row + step * dirx][col + step * diry] is EMPTY for step in range(1,distance))
                                if check_empty_bw:
                                    temp_board[row][col] = EMPTY
                                    temp_board[next_row][next_col] = EMPTY
                                    if jump_row == 0:
                                        temp_board[jump_row][jump_col] = BLACK_RAICHU
                                    else:
                                        temp_board[jump_row][jump_col] = piece
                                    moves.append(temp_board)
                            else:
                                temp_board[row][col] = EMPTY
                                temp_board[next_row][next_col] = EMPTY
                                if jump_row == 0:
                                    temp_board[jump_row][jump_col] = BLACK_RAICHU
                                else:
                                    temp_board[jump_row][jump_col] = piece
                                moves.append(temp_board)  
    return moves

def get_valid_raichu_moves(board, piece, row, col, N):
    moves = []
    directions = [(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)]
    if piece == WHITE_RAICHU:
        for dirx, diry in directions:
            for distance in range(1,N):
                temp_board = copy.deepcopy(board)
                next_row = row + distance * dirx
                next_col = col + distance * diry
                if 0 <= next_row < N and 0 <= next_col < N:
                    if temp_board[next_row][next_col] == EMPTY:
                        check_empty_bw = all(temp_board[row + step * dirx][col + step * diry] is EMPTY for step in range(1,distance))
                        if check_empty_bw:
                            temp_board[row][col] = EMPTY
                            temp_board[next_row][next_col] = piece
                            moves.append(temp_board)
                    elif temp_board[next_row][next_col] in (BLACK_PICHU, BLACK_PIKACHU, BLACK_RAICHU):
                        jump_row = row + (distance + 1) * dirx
                        jump_col = col + (distance + 1) * diry
                        if 0 <= jump_row < N and 0 <= jump_col < N and temp_board[jump_row][jump_col] == EMPTY:
                            check_empty_bw = all(temp_board[row + step * dirx][col + step * diry] is EMPTY for step in range(1,distance))
                            if check_empty_bw:
                                temp_board[row][col] = EMPTY
                                temp_board[next_row][next_col] = EMPTY
                                temp_board[jump_row][jump_col] = piece
                                moves.append(temp_board)
                                while True:
                                    next_row = jump_row + dirx
                                    next_col = jump_col + diry
                                    if 0 <= next_row < N and 0 <= next_col < N and temp_board[next_row][next_col] == EMPTY:
                                        temp_board[jump_row][jump_col] = EMPTY
                                        temp_board[next_row][next_col] = piece
                                        moves.append(temp_board)
                                        jump_row = next_row
                                        jump_col = next_col
                                    else:
                                        break  
                            else:
                                break
    
    elif piece == BLACK_RAICHU:
        for dirx, diry in directions:
            for distance in range(1,N):
                temp_board = copy.deepcopy(board)
                next_row = row + distance * dirx
                next_col = col + distance * diry
                if 0 <= next_row < N and 0 <= next_col < N:
                    if temp_board[next_row][next_col] == EMPTY:
                        check_empty_bw = all(temp_board[row + step * dirx][col + step * diry] is EMPTY for step in range(1,distance))
                        if check_empty_bw:
                            temp_board[row][col] = EMPTY
                            temp_board[next_row][next_col] = piece
                            moves.append(temp_board)
                    elif temp_board[next_row][next_col] in (WHITE_PICHU, WHITE_PIKACHU, WHITE_RAICHU):
                        jump_row = row + (distance + 1) * dirx
                        jump_col = col + (distance + 1) * diry
                        if 0 <= jump_row < N and 0 <= jump_col < N and temp_board[jump_row][jump_col] == EMPTY:
                            check_empty_bw = all(temp_board[row + step * dirx][col + step * diry] is EMPTY for step in range(1,distance))
                            if check_empty_bw:
                                temp_board[row][col] = EMPTY
                                temp_board[next_row][next_col] = EMPTY
                                temp_board[jump_row][jump_col] = piece
                                moves.append(temp_board)
                                while True:
                                    next_row = jump_row + dirx
                                    next_col = jump_col + diry
                                    if 0 <= next_row < N and 0 <= next_col < N and temp_board[next_row][next_col] == EMPTY:
                                        temp_board[jump_row][jump_col] = EMPTY
                                        temp_board[next_row][next_col] = piece
                                        moves.append(temp_board)
                                        jump_row = next_row
                                        jump_col = next_col
                                    else:
                                        break
                            else:
                                break
    return moves

def generate_moves(board, player, N):
    moves = []
    for i in range(N):
        for j in range(N):
            if player == 'w':
                piece = board[i][j]
                if piece == WHITE_PICHU:
                    moves.extend(get_valid_pichu_moves(board, piece, i, j, N))
                elif piece == WHITE_PIKACHU:
                    moves.extend(get_valid_pikachu_moves(board, piece, i, j, N))
                elif piece == WHITE_RAICHU:
                    moves.extend(get_valid_raichu_moves(board, piece, i, j, N))
            elif player == 'b':
                piece = board[i][j]
                if piece == BLACK_PICHU:
                    moves.extend(get_valid_pichu_moves(board, piece, i, j, N))
                elif piece == BLACK_PIKACHU:
                    moves.extend(get_valid_pikachu_moves(board, piece, i, j, N))
                elif piece == BLACK_RAICHU:
                    moves.extend(get_valid_raichu_moves(board, piece, i, j, N))                         
    return moves

def evaluate_board(board, player, N):
    score = 0 

    for i in range(N):
        for j in range(N):
            piece = board[i][j]         
            if piece == WHITE_PICHU:  
                value = 10 + i  
                if player == 'w':
                    score += value
                else:
                    score -= value
            elif piece == WHITE_PIKACHU:  
                value = 30 + i 
                if player == 'w':
                    score += value
                else:
                    score -= value
            elif piece == WHITE_RAICHU: 
                value = 50 
                if player == 'w':
                    score += value
                else:
                    score -= value
            elif piece == BLACK_PICHU: 
                value = 10 + (N - 1 - i) 
                if player == 'b':
                    score += value
                else:
                    score -= value
            elif piece == BLACK_PIKACHU:
                value = 30 + (N - 1 - i) 
                if player == 'b':
                    score += value
                else:
                    score -= value
            elif piece == BLACK_RAICHU:
                value = 50 
                if player == 'b':
                    score += value
                else:
                    score -= value
    return score

def minimax(board, depth, player, N, alpha, beta, ismax):
    if depth == 0:
        return evaluate_board(board, player, N), board
    
    moves = generate_moves(board, player, N)

    if ismax:
        best_score = float('-inf')
        best_move = None
        for move in moves:
            score, _ = minimax(move, depth - 1, 'w' if player == 'b' else 'b', N, alpha, beta, False)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score, best_move
    else:
        best_score = float('inf')
        best_move = None
        for move in moves:
            score, _ = minimax(move, depth - 1, 'w' if player == 'b' else 'b', N, alpha, beta, True)
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score, best_move

def find_best_move(board, N, player, timelimit):
    # This sample code just returns the same board over and over again (which
    # isn't a valid move anyway.) Replace this with your code!
    #
    solvable_board = create_solvable_board(board, N)
    start_time = time.time()
    max_depth = 2
    best_move = None
    prev_score = 0

    while time.time() - start_time < timelimit-1:
        score, move = minimax(solvable_board, max_depth, player, N, float('-inf'), float('inf'), True)
        if best_move is None:
            best_move = move

        if move is not None:
            if score > prev_score:
                prev_score = score
                best_move = move
        max_depth += 2
        res = ''
        for i in range(N):
            for j in range(N):
                res += str(best_move[i][j])
        yield res


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Usage: Raichu.py N player board timelimit")
        
    (_, N, player, board, timelimit) = sys.argv
    N=int(N)
    timelimit=int(timelimit)
    if player not in "wb":
        raise Exception("Invalid player.")

    if len(board) != N*N or 0 in [c in "wb.WB@$" for c in board]:
        raise Exception("Bad board string.")

    print("Searching for best move for " + player + " from board state: \n" + board_to_string(board, N))
    print("Here's what I decided:")
    for new_board in find_best_move(board, N, player, timelimit):
        print(new_board)
