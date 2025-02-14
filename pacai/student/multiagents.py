import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves
        legalMoves = gameState.getLegalActions()
        scores = []
        # Evaluate the best action
        for action in legalMoves:
            score = self.evaluationFunction(gameState, action)
            scores.append(score)
        bestScore = max(scores)
        bestIndices = []
        for index in range(len(scores)):
            if scores[index] == bestScore:
                bestIndices.append(index)
        # Return the best action
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        
        # Calculate closest food distance
        foodList = newFood.asList()
        if len(foodList) > 0:
            foodDistances = [manhattan(newPosition, food) for food in foodList]
            closestFoodDist = min(foodDistances)
        else:
            closestFoodDist = 0
        
        # Calculate ghost distances
        ghostDistances = []
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = manhattan(newPosition, ghostPos)
            if ghostDist < 2 and ghostState.getScaredTimer() == 0:
                return float('-inf')
            ghostDistances.append(ghostDist)
        
        """
        Currently the weights promote pacman to avoid gohst more than get score 
        which causes it to stay on the right side of the board for longer
        """
        score = successorGameState.getScore()
        if closestFoodDist > 0:
            score += 1.0 / closestFoodDist
        score += min(ghostDistances) * 2.0
        
        return score

    def manhattan(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState.
        """
        def minimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return (self.getEvaluationFunction()(state), None)
            
            # Get the legal moves
            legalMoves = state.getLegalActions(agentIndex)
            if agentIndex == 0:  # Pacman's turn (maximizing)
                legalMoves = [a for a in legalMoves if a != 'Stop']
            if not legalMoves:
                return (self.getEvaluationFunction()(state), None)
            
            # Gets the best score 
            if agentIndex == 0:
                bestScore = float('-inf')
                bestAction = None
            else:
                bestScore = float('inf')
                bestAction = None
        
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth
            
            # Evaluation of the best move 
            for action in legalMoves:
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = minimax(successor, nextDepth, nextAgent)
                
                if agentIndex == 0:
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                else:
                    if score < bestScore:
                        bestScore = score
                        bestAction = action
            
            return (bestScore, bestAction)
        
        result = minimax(gameState, self.getTreeDepth(), 0)
        bestScore = result[0]
        bestAction = result[1]
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using alpha-beta pruning.
        """
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return (self.getEvaluationFunction()(state), None)
            
            legalMoves = state.getLegalActions(agentIndex)
            if agentIndex == 0:
                legalMoves = [a for a in legalMoves if a != 'Stop']
            
            if not legalMoves:
                return (self.getEvaluationFunction()(state), None)
            
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth
            
            bestAction = legalMoves[0]
            
            if agentIndex == 0:
                value = float('-inf')
                for action in legalMoves:
                    successor = state.generateSuccessor(agentIndex, action)
                    scoreNew, _ = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                    if scoreNew > value:
                        value = scoreNew
                        bestAction = action
                    alpha = max(alpha, value)
                    if alpha > beta:
                        break
                return (value, bestAction)
            else:
                value = float('inf')
                for action in legalMoves:
                    successor = state.generateSuccessor(agentIndex, action)
                    scoreNew, _ = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                    if scoreNew < value:
                        value = scoreNew
                        bestAction = action
                    beta = min(beta, value)
                    if beta < alpha:
                        break
                return (value, bestAction)
        
        result = alphaBeta(gameState, self.getTreeDepth(), 0, float('-inf'), float('inf'))
        best_action = result[1]
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.getTreeDepth() and self.getEvaluationFunction().
        """
        def expectimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return (self.getEvaluationFunction()(state), None)
            
            legalMoves = state.getLegalActions(agentIndex)
            if agentIndex == 0:
                legalMoves = [a for a in legalMoves if a != 'Stop']
            
            if not legalMoves:
                return (self.getEvaluationFunction()(state), None)
            
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth
            
            if agentIndex == 0:
                bestScore = float('-inf')
                bestAction = legalMoves[0]
                for action in legalMoves:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = expectimax(successor, nextDepth, nextAgent)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                return (bestScore, bestAction)
            else:
                totalScore = 0
                probability = 1.0 / len(legalMoves)
                for action in legalMoves:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = expectimax(successor, nextDepth, nextAgent)
                    totalScore += score * probability
                return (totalScore, legalMoves[0])
    
        bestScore, bestAction = expectimax(gameState, self.getTreeDepth(), 0)
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION -- An evaluation function that scores game states based on:
    1. Current score
    2. Distance to nearest food pellet
    3. Total remaining food pellets
    4. Ghost positions and states (scared vs normal)
    5. Available power pellets

    Higher scores indicate more favorable states
    Lower scores indicate less favorable states
    """
    
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    # init
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()
    
    # Evaluates the food
    foodList = food.asList()
    if len(foodList) > 0:
        foodDistances = [manhattan(pos, food) for food in foodList]
        closestFoodDist = min(foodDistances)
        score += 10.0 / (closestFoodDist + 1)
        score -= len(foodList) * 20
    
    # Evaluation for the gohsts 
    for ghostState in ghostStates:
        ghostDist = manhattan(pos, ghostState.getPosition())
        if ghostState.getScaredTimer() > 0:
            score += 200.0 / (ghostDist + 1)
        else:
            if ghostDist < 2:
                score -= 500
            else:
                score += 2.0 * ghostDist
    
    # Evaluate the capsules 
    numCapsules = len(capsules)
    if numCapsules > 0:
        minGhostDist = min(manhattan(pos, ghost.getPosition()) for ghost in ghostStates)
        if minGhostDist < 4:
            score += 100.0 * numCapsules
        else:
            score += 50.0 * numCapsules
    
    return score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
