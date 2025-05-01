class Action:
    def __init__(self, number: int):
        self.number = number

    def __str__(self):
        return str(self.number)

    def __repr__(self):
        return str(self.number)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.number == other.number
        elif isinstance(other, int):
            return self.number == other
        return False
        
    def __hash__(self):
        return hash(self.number)
