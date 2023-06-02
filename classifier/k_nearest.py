"""
Contains the K-nearest neighbor classifier.
"""
from classifier import base


class KNearestClassifier(base.BaseClassifier):
    """
    Implements the K-nearest neighbor classifier.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.train_data: list[tuple[list[float], int]] = []

    def train(self, x_list: list[list[float]], y_list: list[int]) -> None:
        for i, x_val in enumerate(x_list):
            y_val = y_list[i]
            self.train_data.append((x_val, y_val))

    def predict(self, x_list: list[list[float]]) -> list[int]:
        ans: list[int] = []
        for x_val in x_list:
            # get (distance, y) pairs
            dist_y_pair: list[tuple[float, int]] = []
            for train_x_val, train_y_val in self.train_data:
                dist_y_pair.append(
                    (
                        self.__distance(x_val, train_x_val),
                        train_y_val,
                    )
                )

            # sort
            dist_y_pair.sort(key=lambda x: x[0])

            # count votes for first k in closest distance order
            votes: dict[int, int] = {}
            for i in range(min(self.k, len(dist_y_pair))):
                cur_y = dist_y_pair[i][1]
                if cur_y not in votes:
                    votes[cur_y] = 0
                votes[cur_y] += 1

            # determine winner by number of votes
            max_votes, winner = 0, -1
            for key, num_votes in votes.items():
                if num_votes > max_votes:
                    max_votes = num_votes
                    winner = key
            ans.append(winner)

        return ans

    def __distance(self, x_val1: list[float], x_val2: list[float]) -> float:
        total = 0
        for idx, val1 in enumerate(x_val1):
            val2 = x_val2[idx]
            total += (val1 - val2) ** 2
        return total**0.5
