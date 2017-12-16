# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
from util import manhattanDistance

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        #print(ghosts)
        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        else:
            features["eats-food"] = 0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features


class ourExtractor(SimpleExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        capsules = state.getCapsules()
        capsules_dist = []
        for capsule_i in capsules:
            capsules_dist.append(manhattanDistance(capsule_i,state.getPacmanPosition()))           

        scared = []
        active = []
        scared_dist = []
        active_dist = []
        near_scared = 0
        for i in range(1,len(state.data.agentStates)):
            ghost_state = state.data.agentStates[i]
            ghost_pos = ghost_state.configuration.getPosition()
            if ghost_state.scaredTimer > 0:
                scared.append(ghost_pos)
                if manhattanDistance(ghost_pos,state.getPacmanPosition())<2:
                # print(manhattanDistance(ghost_pos,state.getPacmanPosition()))
                    near_scared = near_scared + 1
                scared_dist.append(manhattanDistance(ghost_pos,state.getPacmanPosition()))
            else:
                active.append(ghost_pos)
                active_dist.append(manhattanDistance(ghost_pos,state.getPacmanPosition()))

        #print(active)
        #print(scared)

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        pos_pacman = [next_x,next_y]

        # count the number of active ghosts 1-step away
        features["#-of-active-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in active)
        #print ((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in active)
        features["#-of-active-ghosts-1or2-step-away"] = sum(len(set(Actions.getLegalNeighbors(pos_pacman,walls)).intersection(Actions.getLegalNeighbors(g,walls))) for g in active)
        #print(features["#-of-active-ghosts-1or2-step-away"])
        # minimum distance to scared ghost
        features["minimum-distance-scared"] = min(scared_dist) if scared_dist else 0

        # count the number of scared ghosts 1-step away
        #features["#-of-scared-ghosts-1-step-away"] = near_scared
        features["#-of-scared-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in scared)
        features["#-of-scared-ghosts-1or2-step-away"] = sum(len(set(Actions.getLegalNeighbors(pos_pacman,walls)).intersection(Actions.getLegalNeighbors(g,walls))) for g in scared)
        #features["#-of-capsules-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(c, walls) for c in capsules)   
        #features["#-of-capsules-1or2-step-away"] = sum(len(set(Actions.getLegalNeighbors(pos_pacman,walls)).intersection(Actions.getLegalNeighbors(c,walls))) for c in capsules)
        if features["#-of-active-ghosts-1or2-step-away"]: 
            features["#-of-capsules-1or2-step-away"]=sum(len(set(Actions.getLegalNeighbors(pos_pacman,walls)).intersection(Actions.getLegalNeighbors(c,walls))) for c in capsules)
            features["minimum-distance-capsule_active"] = min(capsules_dist) if capsules_dist else 0
        else:
            features["#-of-capsules-1or2-step-away"] = 0
            features["minimum-distance-capsule_active"] = 0

        features["dist_active_ghost"] = min(active_dist) if active_dist else 0
        features["minimum-distance-capsule"] = min(capsules_dist) if capsules_dist else 0

        #print(features["#-of-scared-ghosts-1or2-step-away"])

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-active-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        else:
            features["eats-food"] = 0

        #if features["#-of-scared-ghosts-1-step-away"] and food[next_x][next_y]:
            #features["eats-food"] = 0


        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

