# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        value = successorGameState.getScore()
        distanceToGhost = [manhattanDistance(newPos, x.getPosition()) for x in newGhostStates]

        if len(distanceToGhost)>0:
            if min(distanceToGhost) < 2:
                value -= 10000
            else:
                value += min(distanceToGhost)           #longer distance to nearest ghost means higher value

        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            value += 500

        if successorGameState.isWin():
            value += 10000

        distancesToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
        if len(distancesToFood):
            value -= min(distancesToFood)               #longer distance to food means lower value

        return value
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        ghostNum=gameState.getNumAgents()-1
        return self.maxval(gameState, 1, ghostNum)


    def maxval(self, gameState, depth, ghostNum):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        val=float("-inf")
        nextMove=Directions.STOP
        for action in gameState.getLegalActions(0):
            succ=gameState.generateSuccessor(0,action)      #for each successor of state
            nextVal= self.minval(succ,depth,ghostNum,1)        #return max of successor's val
            if nextVal>val:
                val=nextVal
                nextMove=action

        if depth>1:
            return val
        else:
            return nextMove

    def minval(self, gameState, depth, ghostNum, agentIndex):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        val=float("inf")
        if agentIndex==ghostNum:
            for action in gameState.getLegalActions(agentIndex):
                succ = gameState.generateSuccessor(agentIndex, action)
                if depth<self.depth:
                    val = min(val, self.maxval(succ,depth+1,ghostNum))
                else:
                    val = min(val, self.evaluationFunction(succ))
        else:
            for action in gameState.getLegalActions(agentIndex):
                succ = gameState.generateSuccessor(agentIndex, action)
                val = min(val, self.minval(succ,depth,ghostNum,agentIndex+1))
        return val

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ghostNum = gameState.getNumAgents() - 1
        alpha = float("-inf")
        beta = float("inf")
        return self.maxval(gameState, 1, ghostNum, alpha, beta)

    def maxval(self, gameState, depth, ghostNum, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            #beta =
            return self.evaluationFunction(gameState)
        val = float("-inf")
        nextMove = Directions.STOP
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)  # for each successor of state
            nextVal = self.minval(succ, depth, ghostNum, 1, alpha, beta)  # return max of successor's val
            if nextVal > val:
                val = nextVal
                nextMove = action
            if val>beta:
                return val
            alpha = max (val,alpha)

        if depth > 1:
            return val
        else:
            return nextMove

    def minval(self, gameState, depth, ghostNum, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        val = float("inf")
        if agentIndex == ghostNum:
            for action in gameState.getLegalActions(agentIndex):
                succ = gameState.generateSuccessor(agentIndex, action)
                if depth < self.depth:
                    val = min(val, self.maxval(succ, depth + 1, ghostNum, alpha, beta))
                    if val<alpha:
                        return val
                    beta = min(beta, val)
                else:
                    val = min(val, self.evaluationFunction(succ))
                    if val<alpha:
                        return val
                    beta = min(beta, val)
        else:
            for action in gameState.getLegalActions(agentIndex):
                succ = gameState.generateSuccessor(agentIndex, action)
                val = min(val, self.minval(succ, depth, ghostNum, agentIndex + 1, alpha, beta))
                if val < alpha:
                    return val
                beta = min(beta, val)
        return val
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        ghostNum = gameState.getNumAgents() - 1
        return self.maxval(gameState, 1, ghostNum)

    def maxval(self, gameState, depth, ghostNum):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        val = float("-inf")
        nextMove = Directions.STOP
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)  # for each successor of state
            nextVal = self.minval(succ, depth, ghostNum, 1)  # return max of successor's val
            if nextVal > val:
                val = nextVal
                nextMove = action

        if depth > 1:
            return val
        else:
            return nextMove

    def minval(self, gameState, depth, ghostNum, agentIndex):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        val = 0
        p = 1/len(gameState.getLegalActions(agentIndex))
        if agentIndex == ghostNum:
            if depth < self.depth:
                for action in gameState.getLegalActions(agentIndex):
                    succ = gameState.generateSuccessor(agentIndex, action)
                    val += p*self.maxval(succ, depth + 1, ghostNum)
            else:
                for action in gameState.getLegalActions(agentIndex):
                    succ = gameState.generateSuccessor(agentIndex, action)
                    val += p*self.evaluationFunction(succ)
        else:
            for action in gameState.getLegalActions(agentIndex):
                succ = gameState.generateSuccessor(agentIndex, action)
                val += p*self.minval(succ, depth, ghostNum, agentIndex + 1)
        return val

        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    pacmanPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood()
    capPos = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    value = currentGameState.getScore()
    #distanceToGhost = [manhattanDistance(pacmanPos, x.getPosition()) for x in ghostStates]
    if currentGameState.isWin():
        value += 10000

    for ghost in ghostStates:
        distanceToGhost=manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer>0:
            value += 100/distanceToGhost         # if ghost is scared, then shorter distance means higher value
        else:
            if distanceToGhost == 1 :         #very close, means very low value to run away
                value -= 1000
            else:
                value += distanceToGhost     # longer distance to ghost means higher value

    '''
    if len(distanceToGhost) > 0:
        if min(distanceToGhost) < 2:
            value -= 10000
        else:
            value += min(distanceToGhost)  # longer distance to nearest ghost means higher value
    '''

    distancesToFood = [manhattanDistance(pacmanPos, x) for x in foodPos.asList()]
    if len(distancesToFood):
        value += 10/min(distancesToFood)  # longer distance to food means lower value

    return value

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
