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
