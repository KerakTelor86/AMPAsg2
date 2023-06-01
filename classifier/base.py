"""
Contains the base classifier.
"""
import abc


class BaseClassifier(abc.ABC):
    """
    Base classifier signature to inherit from.
    """

    @abc.abstractmethod
    def train(self, x_list: list[list[float]], y_list: list[int]) -> None:
        """
        Trains the classifier using the provided data.
        """
        return

    @abc.abstractmethod
    def predict(self, x_list: list[list[float]]) -> list[int]:
        """
        Returns a prediction for every X in the provided list.
        """
        return []
