from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    """
    abstrakt grensesnitt for modeller som brukes i tree-search algoritmer
    dette lar oss bytte mellom perfekte simulator-modeller (for MCTS) 
    og lærte modeller (for MuZero)
    """
    
    @abstractmethod
    def represent_state(self, observation):
        """
        konverterer en faktisk observasjon fra miljøet til en abstrakt tilstandsrepresentasjon
        
        args:
            observation: den faktiske observasjonen fra miljøet
            
        returns:
            en abstrakt tilstandsrepresentasjon
        """
        pass
    
    @abstractmethod
    def transition(self, abstract_state, action):
        """
        forutsier neste abstrakte tilstand og belønning etter å ha utført en handling
        
        args:
            abstract_state: den nåværende abstrakte tilstanden
            action: handlingen som skal utføres
            
        returns:
            (next_abstract_state, reward)
        """
        pass
    
    @abstractmethod
    def predict(self, abstract_state):
        """
        forutsier policy og value for en gitt abstrakt tilstand
        
        args:
            abstract_state: den abstrakte tilstanden som skal evalueres
            
        returns:
            (policy, value) hvor:
            - policy er en sannsynlighetsfordeling over handlinger
            - value er den forventede fremtidige diskonterte avkastningen fra denne tilstanden
        """
        pass