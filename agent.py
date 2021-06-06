"""
My heuristic consider 3 components: 
1. coin parity, I will use compute_utility directly here.
2. the difference between the number of "my" possible moves and the opponent's possible moves.
3. the difference of "my" stable discs and the opponent's stable discs. A stable disc cannot be flipped by the other player in the next round.
Occupied discs on the edges and on the corners will receive extra stability bonus since they are quite stable. 
I created a helper function count_stable_discs to implement this, which is a modification on the find_lines function.
My heuristic is a linear combination of the 3 components. The coefficients are given based on the importance of each component.
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move
min_dict = dict()
max_dict = dict()
alpha_dict = dict()
beta_dict = dict()

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
# Method to compute utility value of terminal state
def compute_utility(board, color):
    #IMPLEMENT
    final_score = get_score(board)
    if color == 1:
        return final_score[0] - final_score[1]

    return final_score[1] - final_score[0]

def opponent(player):
    if player == 1:
        return 2
    else: return 1

# Better heuristic value of board
def count_stable_discs(board, player):
#count how many discs CANNOT be flipped by the other player
    stable_discs = 0
    other_player = opponent(player)
    
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == player:
                #corners and edges stability bonus
                if i == 0:
                    stable_discs += 0.5 #edge
                    if j == 0 or j == len(board)-1:
                        stable_discs += 1 #corner
                elif j == 0:
                    stable_discs += 0.5 #edge
                    if i == len(board)-1:
                        stable_discs += 1 #corner             

                unstable = False
                for xdir, ydir in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
                    u = i
                    v = j
                    u += xdir
                    v += ydir
                    while u >= 0 and u < len(board) and v >= 0 and v < len(board):
                        if board[v][u] == 0:
                            break
                        elif board[v][u] == other_player:
                            unstable = True
                            break
                        else: 
                            u += xdir
                            v += ydir
                if unstable == False:
                    stable_discs += 1
            
    return stable_discs


def compute_heuristic(board, color): 
    oppo = opponent(color)
    #number of available moves difference
    mob = len(get_possible_moves(board, color)) - len(get_possible_moves(board, oppo))
    #number of stable discs difference
    stab = count_stable_discs(board, color) - count_stable_discs(board, oppo)
    #coin parity
    cp = compute_utility(board, color)
        
    return 0.4*cp + 0.25*mob + 0.35*stab
    
############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0):
    #color of Min player
    oppo_color = opponent(color)
    if caching and board in min_dict:
        return min_dict[board]
            
    moves = get_possible_moves(board, oppo_color)
    if moves == [] or limit == 0:
        result = (None, compute_utility(board, color))#utility from Max's position
        return result

    min_util = float("inf")
    best_move = moves[0]

    for move in moves:
        new_board = play_move(board, oppo_color, move[0], move[1])
        if caching and (new_board in min_dict):
            util = min_dict[new_board][1]
        else:
            util = minimax_max_node(new_board, color, limit-1, caching)[1]
            if util < min_util:
                best_move = move
                min_util = util

    if caching:
        min_dict[board] = (best_move, min_util)
        
    return (best_move, min_util)

def minimax_max_node(board, color, limit, caching = 0):  
    if caching:
        if board in max_dict:
            return max_dict[board]

    moves = get_possible_moves(board, color)
    if moves == [] or limit == 0:
        result = (None, compute_utility(board, color))
        return result

    max_util = float("-inf")
    best_move = moves[0]

    for move in moves:
        new_board = play_move(board, color, move[0], move[1])
        if caching and (new_board in max_dict):
            util = max_dict[new_board][1]
        else:
            util = minimax_min_node(new_board, color, limit-1, caching)[1]
            if util > max_util:
                best_move = move
                max_util = util

    if caching:
        max_dict[board] = (best_move, max_util)
    
    return (best_move, max_util)

def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    #IMPLEMENT
    return minimax_max_node(board, color, limit, caching)[0]

############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    # Get the color of the other player
    oppo_color = opponent(color)
    if caching:
        if board in beta_dict:
            return beta_dict[board]
    moves = get_possible_moves(board, oppo_color)

    if moves == [] or limit == 0:
        result = (None, compute_utility(board, color))
        return result
    else:  
        # Get the maximum utility 
        min_utility = float("inf")
        min_move = None
        all_moves = []
        for move in moves:
            new_board = play_move(board, oppo_color, move[0], move[1])
            all_moves.append((move, new_board))

        # Sort by Max player's utility from small to large
        if ordering:
            all_moves.sort(key = lambda util: compute_utility(util[1], color), reverse = False)

        for move in all_moves:
            if caching and (move[1] in beta_dict):
                best_move, beta_util = beta_dict[move[1]]
            else:
                best_move, beta_util = alphabeta_max_node(move[1], color, alpha, beta, limit-1, caching, ordering)
                if caching:
                    beta_dict[move[1]] = (best_move, beta_util)

            if beta_util < min_utility:
                min_utility = beta_util
                min_move = move[0]

            if min_utility <= alpha:
                return min_move, min_utility 

            if min_utility < beta:
                beta = min_utility
                
        if caching:
            beta_dict[board] = (min_move, min_utility)

        return min_move, min_utility


def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
            
    # Get the allowed moves
    moves = get_possible_moves(board, color)

    if moves == [] or limit == 0:
        result = (None, compute_utility(board, color))
        return result
    else:  

        max_utility = float("-inf")
        max_move = None
        moves_sorted = []

        for move in moves:
            new_board = play_move(board, color, move[0], move[1])
            moves_sorted.append((move, new_board)) 
        # Sort by Max player's utility in reverse
        if ordering:
            moves_sorted.sort(key = lambda util: compute_utility(util[1], color), reverse=True)
            
        for move in moves_sorted:
            if caching and (move[1] in alpha_dict):
                best_move, alpha_util = alpha_dict[move[1]]
            else:
                best_move, alpha_util = alphabeta_min_node(move[1], color, alpha, beta, limit-1, caching, ordering)
                if caching:
                    alpha_dict[move[1]] = (best_move, alpha_util)

            if alpha_util > max_utility:
                max_utility = alpha_util
                max_move = move[0]

            if max_utility >= beta:
                return max_move, max_utility

            if max_utility > alpha:
                alpha = max_utility
        if caching:
            alpha_dict[board] = (max_move, max_utility)
  
        return max_move, max_utility


def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    #IMPLEMENT
    return alphabeta_max_node(board, color, float("-inf"), float("inf"), limit, caching, ordering)[0] 

####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching 
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)
                                  
            #TA said we can add this line
            board = tuple(tuple(row) for row in board)
            
            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
