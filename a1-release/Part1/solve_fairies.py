#!/usr/local/bin/python3
# solve_fairies.py : Fairy puzzle solver
#
# Code by: Prinston Rebello (prebello@iu.edu) and Prachi Jethava(pjethava@iu.edu)
#
# Based on skeleton code by B551 course staff, Fall 2023
#
# N fairies stand in a row on a wire, each adorned with a magical symbol from 1 to N.
# In a single step, two adjacent fairies can swap places. How can
# they rearrange themselves to be in order from 1 to N in the fewest
# possible steps?

# !/usr/bin/env python3
import sys
from queue import PriorityQueue

N=5

#####
# THE ABSTRACTION:
#
# Initial state:

# Goal state:
# given a state, returns True or False to indicate if it is the goal state
def is_goal(state):
    return state == list(range(1, N+1))

# Successor function:
# given a state, return a list of successor states
def successors(state):
    return [ state[0:n] + [state[n+1],] + [state[n],] + state[n+2:] for n in range(0, N-1) ]

# Heuristic function:
# given a state, return an estimate of the number of steps to a goal from that state
def h(current_state, goal_state):
    misplaced_moves = 0
    for i in range(0, N):
         if current_state[i] != goal_state[i]:
            misplaced_moves += 1
    return misplaced_moves

#########
#
# THE ALGORITHM:
#
# This is a generic solver using BFS. 
#
def solve(initial_state):
    goal_state = sorted(initial_state)
    fringe = PriorityQueue()
    path = []
    fringe.put((h(initial_state, goal_state), initial_state, path))
    visited = []
    while fringe.qsize() > 0:
        _, current_state, path = fringe.get()
        visited.append(current_state)

        if is_goal(current_state):
            return path+[current_state,]

        for s in successors(current_state):
            if s not in visited:
                fringe.put((h(s, goal_state), s, path+[current_state,]))

    return []

# Please don't modify anything below this line
#
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: expected a test case filename"))
    test_cases = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            test_cases.append([ int(i) for i in line.split() ])
    i = 0
    for initial_state in test_cases:
        	print('From state ' + str(initial_state) + " found goal state by taking path: " + str(solve(initial_state)))


    

