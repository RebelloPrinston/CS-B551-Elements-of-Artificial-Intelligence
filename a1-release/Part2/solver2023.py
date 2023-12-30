#!/usr/local/bin/python3
# solver2023.py : 2023 Sliding tile puzzle solver
#
# Code by: Prinston Rebello (prebello@iu.edu) abd Prachi Jethava (pjethava@iu.edu)
#
# Based on skeleton code by B551 Staff, Fall 2023
#

import sys
from queue import PriorityQueue

ROWS=5
COLS=5

def printable_board(board):
    return [ ('%3d ')*COLS  % board[j:(j+COLS)] for j in range(0, ROWS*COLS, COLS) ]

def create_solvable_board(board):
    tracker = 0
    transformed_board = [[0] * 5 for _ in range(5)]
    for i in range(ROWS):
        for j in range(COLS):
            transformed_board[i][j] = board[tracker]
            tracker += 1
    return transformed_board

def create_goal_state():
    goal_state = []
    num = 1
    for _ in range(5):
        row = []
        for _ in range(5):
            row.append(num)
            num += 1
        goal_state.append(row)
    return goal_state


def move_left_or_right(current_board, row, row_num, move):
    if move == 'L':
        row.append(row.pop(0))
    elif move == 'R':
        row.insert(0, row.pop())
    
    current_board[row_num] = row
    return current_board

def move_up_or_down(current_board, column, move):
    col = []
    for i in range(ROWS):
        col.append(current_board[i][column])
    
    if move == 'U':
        col.append(col.pop(0))
    elif move == 'D':
        col.insert(0, col.pop())

    for i in range(ROWS):
        current_board[i][column] = col[i]
    
    return current_board

# https://www.geeksforgeeks.org/rotate-matrix-elements/#
def outer_rotation(current_board, move):
    top = 0
    bottom = ROWS - 1
    left = 0
    right = COLS - 1
    if move == 'Oc':
        temp_element = current_board[top + 1][left]
        for i in range(left, right + 1):
            current_element = current_board[top][i]
            current_board[top][i] = temp_element
            temp_element = current_element
        top += 1
        
        for i in range(top, bottom + 1):
            current_element = current_board[i][right]
            current_board[i][right] = temp_element
            temp_element = current_element
        right -= 1
        
        for i in range(right, left - 1, -1):
            current_element = current_board[bottom][i]
            current_board[bottom][i] = temp_element
            temp_element = current_element
        bottom -= 1

        for i in range(bottom, top - 1, -1):
            current_element = current_board[i][left]
            current_board[i][left] = temp_element
            temp_element = current_element
        left += 1
    
    elif move == 'Occ':
        temp_element = current_board[top][left + 1]    
        for i in range(top, bottom + 1):
            current_element = current_board[i][left]
            current_board[i][left] = temp_element
            temp_element = current_element
        left += 1
        
        for i in range(left, right + 1):
            current_element = current_board[bottom][i]
            current_board[bottom][i] = temp_element
            temp_element = current_element
        bottom -= 1
            
        for i in range(bottom, top - 1, -1):
            current_element = current_board[i][right]
            current_board[i][right] = temp_element
            temp_element = current_element
        right -= 1
        
        for i in range(right, left - 1, -1):
            current_element = current_board[top][i]
            current_board[top][i] = temp_element
            temp_element = current_element
        top += 1

    return current_board

def inner_rotation(current_board, move):
    top = 1
    bottom = ROWS - 2
    left = 1
    right = COLS - 2
    if move == 'Ic':
        temp_element = current_board[top + 1][left]
        for i in range(left, right + 1):
            current_element = current_board[top][i]
            current_board[top][i] = temp_element
            temp_element = current_element
        top += 1
        
        for i in range(top, bottom + 1):
            current_element = current_board[i][right]
            current_board[i][right] = temp_element
            temp_element = current_element
        right -= 1
        
        for i in range(right, left - 1, -1):
            current_element = current_board[bottom][i]
            current_board[bottom][i] = temp_element
            temp_element = current_element
        bottom -= 1

        for i in range(bottom, top - 1, -1):
            current_element = current_board[i][left]
            current_board[i][left] = temp_element
            temp_element = current_element
        left += 1

    elif move == 'Icc':
        temp_element = current_board[top][left + 1]    
        for i in range(top, bottom + 1):
            current_element = current_board[i][left]
            current_board[i][left] = temp_element
            temp_element = current_element
        left += 1
        
        for i in range(left, right + 1):
            current_element = current_board[bottom][i]
            current_board[bottom][i] = temp_element
            temp_element = current_element
        bottom -= 1
            
        for i in range(bottom, top - 1, -1):
            current_element = current_board[i][right]
            current_board[i][right] = temp_element
            temp_element = current_element
        right -= 1
        
        for i in range(right, left - 1, -1):
            current_element = current_board[top][i]
            current_board[top][i] = temp_element
            temp_element = current_element
        top += 1

    return current_board

def find_tile(state, element):
    for i in range(ROWS):
        for j in range(COLS):
            if state[i][j] == element:
                return i, j
    return -1,-1


# return a list of possible successor states
def successors(state, path):
    current_state = state
    successor_set = []
    for row in range(ROWS):
        for move in ('L','R'):
            temp_path = path + [move + str(row + 1)]
            new_state = [list(rows) for rows in current_state]
            new_state = move_left_or_right(new_state, new_state[row], row, move)
            successor_set.append((new_state, temp_path))

    for column in range(COLS):    
        for move in ('U','D'):
            temp_path = path + [move + str(column + 1)]
            new_state =[list(rows) for rows in current_state]
            new_state = move_up_or_down(new_state, column, move)
            successor_set.append((new_state, temp_path))

    for move in ('Oc','Occ'):
        temp_path = path + [move]
        new_state = [list(row) for row in current_state]
        new_state = outer_rotation(new_state, move)
        successor_set.append((new_state, temp_path))
    
    for move in ('Ic','Icc'):
        temp_path = path + [move]
        new_state = [list(row) for row in current_state]
        new_state = inner_rotation(new_state, move)
        successor_set.append((new_state, temp_path))
    return successor_set

# check if we've reached the goal
def is_goal(state, goal_state):
    for i in range(ROWS):
        for j in range(COLS):
            if state[i][j] != goal_state[i][j]:
                return False
    return True

def h(state, goal_state):
    total_distance = 0
    for i in range(ROWS):
        for j in range(COLS):
            tile = state[i][j]
            goal_row, goal_col = find_tile(goal_state, tile)
            total_distance += abs(i - goal_row) + abs(j - goal_col)
    return total_distance*0.2
 

def solve(initial_board):
    """
    1. This function should return the solution as instructed in assignment, consisting of a list of moves like ["R2","D2","U1"].
    2. Do not add any extra parameters to the solve() function, or it will break our grading and testing code.
       For testing we will call this function with single argument(initial_board) and it should return 
       the solution.
    3. Please do not use any global variables, as it may cause the testing code to fail.
    4. You can assume that all test cases will be solvable.
    5. The current code just returns a dummy solution.
    """
    visited = []
    current_board = create_solvable_board(initial_board)
    goal_state = create_goal_state()
    # https://stackoverflow.com/questions/35004882/make-a-list-of-ints-hashable-in-python
    priority_queue = PriorityQueue()
    priority_queue.put((h(current_board, goal_state), current_board, []))
    while priority_queue.qsize() > 0:
        _, current_state, path = priority_queue.get()

        if is_goal(current_state, goal_state):
            return path
        
        visited.append(current_state)
        
        #print(successors(current_state))
        g = len(path)
        for next_state, next_path in successors(current_state, path):
            if next_state not in visited:
                priority_queue.put((h(next_state, goal_state) + g, next_state, next_path))

    return []

# Please don't modify anything below this line
#
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: expected a board filename"))

    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state += [ int(i) for i in line.split() ]

    if len(start_state) != ROWS*COLS:
        raise(Exception("Error: couldn't parse start state file"))

    print("Start state: \n" +"\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    route = solve(tuple(start_state))
    
    print("Solution found in " + str(len(route)) + " moves:" + "\n" + " ".join(route))
