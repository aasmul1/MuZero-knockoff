from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from project2.Action import Action


class Model(ABC):
    """
    Abstrakt grensesnitt for modeller som brukes i tree-search algoritmer.
    Dette lar oss bytte mellom perfekte simulator-modeller (for MCTS) 
    og lærte modeller (for MuZero).
    """

    @abstractmethod
    def represent_state(self, observation: np.ndarray) -> np.ndarray:
        """
        Konverterer en faktisk observasjon fra miljøet til en abstrakt tilstandsrepresentasjon.
        
        Args:
            observation: Den faktiske observasjonen fra miljøet
            
        Returns:
            En abstrakt tilstandsrepresentasjon
        """
        pass

    @abstractmethod
    def transition(self, abstract_state: np.ndarray, action: Action) -> Tuple[np.ndarray, float]:
        """
        Forutsier neste abstrakte tilstand og belønning etter å ha utført en handling.
        
        Args:
            abstract_state: Den nåværende abstrakte tilstanden
            action: Handlingen som skal utføres
            
        Returns:
            (next_abstract_state, reward)
        """
        pass

    @abstractmethod
    def predict(self, abstract_state: np.ndarray) -> Tuple[Dict[Action, float], float]:
        """
        Forutsier policy og value for en gitt abstrakt tilstand.
        
        Args:
            abstract_state: Den abstrakte tilstanden som skal evalueres
            
        Returns:
            (policy, value) hvor:
            - policy er en sannsynlighetsfordeling over handlinger
            - value er den forventede fremtidige diskonterte avkastningen fra denne tilstanden
        """
        pass
