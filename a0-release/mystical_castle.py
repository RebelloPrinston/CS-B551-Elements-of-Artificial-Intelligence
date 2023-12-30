#!/usr/local/bin/python3
#
# mystical_castle.py : a maze solver
#
# Submitted by : Prinston Rebello (prebello@iu.edu)
#
# Based on skeleton code provided in CSCI B551, Fall 2023.

import sys

# Parse the map from a given filename
def parse_map(filename):
        with open(filename, "r") as f:
                return [[char for char in line] for line in f.read().rstrip("\n").split("\n")][3:]
                
# Check if a row,col index pair is on the map
def valid_index(pos, n, m):
        return 0 <= pos[0] < n  and 0 <= pos[1] < m

# Find the possible moves from position (row, col)
def moves(map, row, col):
        moves=((row+1,col), (row-1,col), (row,col-1), (row,col+1))

        # Return only moves that are within the castle_map and legal (i.e. go through open space ".")
        return [ move for move in moves if valid_index(move, len(map), len(map[0])) and (map[move[0]][move[1]] in ".@" ) ]

# Perform search on the map
#
# This function MUST take a single parameter as input -- the castle map --
# and return a tuple of the form (move_count, move_string), where:
# - move_count is the number of moves required to navigate from start to finish, or -1
#    if no such route exists
# - move_string is a string indicating the path, consisting of U, L, R, and D characters
#    (for up, left, right, and down)
def getDirection(current_move, next_move):
       if current_move[0] < next_move[0] and current_move[1] == next_move[1]: 
              return 'D'
       elif current_move[0] > next_move[0] and current_move[1] == next_move[1]: 
              return 'U'
       elif current_move[1] < next_move[1] and current_move[0] == next_move[0]: 
              return 'R'
       elif current_move[1] > next_move[1] and current_move[0] == next_move[0]:
              return 'L'
        

def search(castle_map):
        # Find current start position
        # Reference: https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/#
        
        current_loc=[(row_i,col_i) for col_i in range(len(castle_map[0])) for row_i in range(len(castle_map)) if castle_map[row_i][col_i]=="p"][0]
        visited = []
        fringe=[(current_loc,0,'')]
        while fringe:
                (curr_move, curr_dist, path)=fringe.pop(0)
                for move in moves(castle_map, *curr_move):
                        if castle_map[move[0]][move[1]]=="@":
                                return (curr_dist + 1, path + getDirection(curr_move, move))
                        else:   
                            if move not in visited:
                                visited.append(move)
                                fringe.append((move, curr_dist + 1, path + getDirection(curr_move, move)))

        return (-1,'')

# Main Function
if __name__ == "__main__":
        castle_map=parse_map(sys.argv[1])
        print("Shhhh... quiet while I navigate!")
        solution = search(castle_map)
        print("Here's the solution I found:")
        print(str(solution[0]) + " " + solution[1])
        # print (len(castle_map))
        # print (len(castle_map[0]))
        # print(castle_map)

