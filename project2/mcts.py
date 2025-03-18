import copy
import time
import numpy as np
import random

class Node():
    def __init__(self, parent, game, action, observation, done):
        self.parent = parent
        self.game = game
        self.action = action
        self.observation = observation
        self.done = done
        self.children = []
        self.visits = 0
        self.reward = 0
        
    def getUCBScore(self):
        
        c = 1.41
        
        if self.visits == 0:
            return float('inf')
        
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
            
        return (self.reward / self.visits) + c * np.sqrt(np.log(top_node.visits) / self.visits) 
        
    def create_child(self):
        if self.done:
            return None
        
        actions = []
        games = []
        
        for i in range(self.game.actions()):
            actions.append(i)
            game.append(copy.deepcopy(self.game))
            
        child = {}
        
        '''For each pair, it performs a step in the 
        game using the action, then creates a new Node 
        that captures the resulting game state (observation), 
        reward, and done flag, and stores this Node as a child 
        associated with that action'''
        
        for action, game in zip(actions, games):
            observation, reward, done, _ = game.step(action)
            child[action] = Node(game, done, self, observation, action)                        
            
        self.child = child
        
    def explore(self):
        
        current_node = self
        while current_node.children:
            child = current_node.children
            max_U = max(c.getUBCScore() for c in child.values())
            actions = [ a for a,c in child.items() if c.getUBCScore == max_U]
            if len(actions)==0:
                print("error zero length", max_U)
            action = random.choice(actions)
            current_node = child[action]
            
        if current_node.visits < 1:
            current_node.reward = current_node.reward + current_node.rollout()
        else: 
            current_node.create_child()
            if current_node.children:
                current_node = random.choice(current_node.children)
            current_node.reward = current_node.reward + current_node.rollout()
            
        current_node.visits += 1
        
        while current_node.parent:
                    
            child_reward = current_node.reward
            current_node = current_node.parent
            current_node.visits += 1
            current_node.reward += child_reward             
            
       
    
    def rollout(self):
        
        if self.done:
            return 0
        
        v = 0
        done = False
        new_game = copy.deepcopy(self.game)
        while not done:
            #get a random action
            action = new_game.action_space.sample()
            observation, reward, done, _ = new_game.step(action)
            v = v + reward
            if done:
                new_game.reset()
                new_game.close()
                break
        return v
    
    def next(self):
        if self.done:
            raise ValueError("Game has ended")
        
        if not self.children:
            raise ValueError("No children found and game hasn\'t ended")
        
        child = self.children
        
        max_visits = max(node.visits for node in child.values())

        max_children = [ c for a,c in child.items() if c == max_visits]
        
        if len(max_children) == 0:
            print("Error xero length ", max_visits)
            
        max_child = random.choice(max_children)
        
        return max_child, max_child.action
    
    
    def MCTS(mytree):
        MCTS_POLICY_EXPLORE = 100

        for i in range(MCTS_POLICY_EXPLORE):
            mytree.explore()
            
        next_tree, next_action = mytree.next()
        
        
        next_tree.detatch_parent()
        
        return next_tree, next_action
    
    