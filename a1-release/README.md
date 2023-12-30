# Question Part 1

## The main idea is to implement the queue into priority queue and using the cost function f(s)= g(s) + h(s)

## Question: we can only swap 1 fairy and need to arrange them in the ascending order.

## Priority Queue

- Priority values are given to the elements.
- The element with higher pririty value is retrived first than the other one
- Priority queue works on the basis of the stack
- When the element is inserted it will be inserted on the position on basis of the priority value

# Question Part 2

## Question: Need to arrange the matrix of 5x5 into the order from 1 to 25 from the operations i.e. L, R, U, D, OC, OCC, ICC, IC in the minimul amount of time possible.

## The main idea to implement this is the Manhattan distance with the Priority Queue with the A\* algorithm as mentioned.

### Approach used previously

- Misplaced tiles approach.

### Problem Faced:

- The function initially was going in infinite.
- The function was overestimating

### Resolving the previous approach

- used the approach of the calculating the manhattan distance.
- for the overestimation multiplied the function with 0.2 thus to overcome the infinite loop and overestimation.

## What is the branching factor of the search tree.

- The main idea of the branching factor is that how many moves are possible from that particular current state.
- Here we have different moves as that a particular can take that are L, R, ICC,IC, OCC, OC, U, D.

- So for each left and right, every row has 2 possible outcome that is left and right so branching factor would be 2\*row.

- Same goes with the column, it has two option either go up or down. So the branching factor would be 2\* column.

- So there are total 5 rows and 5 columns thus 2^5 and 2^5

- There are 2 ppossilbe ways for the circular rotation so 2 for inner loop and 2 for outer loop.

- Thus the total branching factor would be 2^5 + 2^5 +2 +2= 68

## If the solution can be reached in 7 moves, about how many states would we need to explore before we found it if we used BFS instead of A\* search?

- we know that that the total branching facotr is 68 as calulated above

- The functioning of the BFS is it will traverse all the possible path from that given point.

- BFS will give all the possible moves from that place till the end.

- So according to that concept the BFS approach will have the total possible moves is 68 and there are 7 moves let assume according to the data given in the question.

- Thus the total BFS moves would be 68^7

- This number is huge and will take a lot of time to reach there which is not feasible approach.

- As we know the A\* is a informed search algorithm which is combinition of BFS and DFS.

- In the given solution we have used the Heuristic approach and priority queue which will reduce the number of traversal from the current node.

- Thus the A\* algorithm is the better approach to solve this question than BFS.
