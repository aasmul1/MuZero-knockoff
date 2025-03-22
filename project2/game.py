from abc import ABC, abstractmethod


class Game(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

    @abstractmethod
    def copy(self) -> "Game":
        pass

    @abstractmethod
    def is_terminal_state(self):
        pass

    @abstractmethod
    def get_legal_actions(self):
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def close(self):
        pass

    # TODO Difference between apply and step?
    @abstractmethod
    def apply(self, action):
        pass
