"""Helper module used for handling results."""
from dataclasses import dataclass


@dataclass
class Best:
    """A class used to represent the best result of an evaluation.

    Attributes:
        accuracy (float): Best accuracy achieved.
        alpha (float): Alpha value associated with the best accuracy.
        beta (float): Beta value associated with the best accuracy.
        gamma (float): Gamma value associated with the best accuracy.
    """

    accuracy: float = 0
    alpha: float = 0
    beta: float = 0
    gamma: float = 0
