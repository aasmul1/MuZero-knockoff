import copy
import time
import numpy as np
import random

class Node():
    def __init__(self, parent, sim_game, action):
        self.parent = parent
        self.sim_game = sim_game
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0
        
        
    def copy(self):
        return copy.deepcopy(self)
    
    def follow_tree_policy(self):
        if self.visits == 0:
            return self
        
        best_child = self.get_best_child()
        while True:
            if (best_child.visits == 0 or best_child.sim_game.is_terminal_state() or len(best_child.children) == 0):
                break
            best_child = best_child.get_best_child()
            
        return best_child
        
    def get_best_child(self):
        if not self.children:
            raise ValueError("No children found")
        
        children_scores = [ (child, child.getUBCScore()) for child in self.children if child.visits > 0]
        
        if not children_scores:
            return random.choice(self.children)
        
        max_value = max(children_scores, key=lambda x: x[1])[1]
        
        best_children = [ child for child, score in children_scores if score==max_value ]
        return random.choice(best_children) if best_children else random.choice(self.children)
    
    def expand(self):
        if self.sim_game.is_terminal_state():
            return self
        
        #enten metode fra game_state_manager eller catch.py
        actions = self.sim_game.get_legal_actions()
        if len(self.actions) == 0:
            print("No actions found")
            return self
        
        children = [ Node(self, self.sim_game.copy().apply(action), action)
                    for action in actions if action is not None]
        
        random.shuffle(children)
        self.children = children
        return self.get_best_child()
        
    def getUCBScore(self):
        
        c = 1.41
        
        if self.visits == 0:
            return float('inf')
        
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
            
        return (self.reward / self.visits) + c * np.sqrt(np.log(top_node.visits) / self.visits) 
        
    def backpropagate(self, reward):
        self.visits += 1
        self.value += (reward - self.value)/self.visits
        if self.parent:
            self.parent.backpropagate(reward)
            
            
            
    def get_distribution(self):
        children_action_indices = []
        for child in self.children:
            children_action_indices.append(child.action.get_action_index())
            
        distribution = np.zeros(len(self.sim_game.get_actions()))
        distribution[children_action_indices] = [ child.visits for child in self.children]
        distribution /= distribution.sum()
        return distribution
        
        
class MCTS:
    def __init__(self, sim_game, current_node):
        self.sim_game = sim_game.copy()
        if current_node:
            self.current_node = current_node
        else:
            Node(None, self.sim_game, None)
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
    
    