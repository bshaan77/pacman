"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    
    frontier = Stack()
    frontier.push((problem.startingState(), []))
    visited = set()

    while not frontier.isEmpty():

        current_state, actions = frontier.pop()
        
        if current_state in visited:
            continue

        visited.add(current_state)
        
        if problem.isGoal(current_state):
            return actions
            
        for next_state, action, cost in problem.successorStates(current_state):
            if next_state not in visited:
                frontier.push((next_state, actions + [action]))
    
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    frontier = Queue()
    frontier.push((problem.startingState(), []))
    visited = set()
    
    while not frontier.isEmpty():
        current_state, actions = frontier.pop()
        
        if current_state in visited:
            continue
            
        visited.add(current_state)
        
        if problem.isGoal(current_state):
            return actions
            
        for next_state, action, cost in problem.successorStates(current_state):
            if next_state not in visited:
                frontier.push((next_state, actions + [action]))
    
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    frontier = PriorityQueue()
    frontier.push((problem.startingState(), [], 0), 0)
    visited = set()
    
    while not frontier.isEmpty():
        current_state, actions, total_cost = frontier.pop()
        
        if current_state in visited:
            continue
            
        visited.add(current_state)
        
        if problem.isGoal(current_state):
            return actions
            
        for next_state, action, step_cost in problem.successorStates(current_state):
            if next_state not in visited:
                cost = total_cost + step_cost
                frontier.push((next_state, actions + [action], cost), cost)
    
    return []

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    frontier = PriorityQueue()
    frontier.push((problem.startingState(), [], 0), heuristic(problem.startingState(), problem))
    visited = set()
    
    while not frontier.isEmpty():
        current_state, actions, total_cost = frontier.pop()
        
        if current_state in visited:
            continue
            
        visited.add(current_state)
        
        if problem.isGoal(current_state):
            return actions
            
        for next_state, action, step_cost in problem.successorStates(current_state):
            if next_state not in visited:
                cost = total_cost + step_cost
                frontier.push((next_state, actions + [action], cost),
                              cost + heuristic(next_state, problem))
    
    return []
