from dataclasses import dataclass


@dataclass
class Best:
    accuracy: float = 0
    alpha: float = 0
    beta: float = 0


@dataclass
class Result:
    avg_accuracy: float = 0
    alpha: float = 0
    beta: float = 0


class Results:
    def __init__(self):
        self.results = list()

    def save(self):
        pass

    def load(self):
        pass

    def top_x(self, x=3):
        pass
