"""
Contains Fisher's Linear Discriminant Analysis-based classifier.
"""
import numpy as np
from scipy.stats import norm

from classifier import base


class FisherLDAClassifier(base.BaseClassifier):
    """
    Implements Fisher's Linear Discriminant Analysis-based classifier.
    """

    def __init__(self):
        self.num_classes = 0
        self.optimal_lines: list[np.ndarray] = []
        self.means: list[list[float]] = []
        self.variances: list[list[float]] = []

    def train(self, x_list: list[list[float]], y_list: list[int]) -> None:
        # assuming range of y_data is 0 <= val < num_classes
        for y_data in y_list:
            self.num_classes = max(self.num_classes, y_data)
        self.num_classes += 1

        for i in range(self.num_classes):
            cur_line = _get_optimal_line(x_list, y_list, i)
            self.optimal_lines.append(cur_line)

            split_i: list[list[float]] = [[], []]
            for idx, x_data in enumerate(x_list):
                split_i[y_list[idx] == i].append(np.dot(cur_line, x_data))

            self.means.append([np.mean(x) for x in split_i])
            self.variances.append([np.var(x) for x in split_i])

    def predict(self, x_list: list[list[float]]) -> list[int]:
        result: list[int] = []
        for x_data in x_list:
            result.append(
                np.argmax(
                    [
                        self._get_probability(x_data, i)
                        for i in range(self.num_classes)
                    ]
                )
            )
        return result

    def _get_probability(self, x_data: list[float], y_data: int) -> float:
        value = np.dot(x_data, self.optimal_lines[y_data])
        probabilities = [
            norm.logpdf(
                (value - self.means[y_data][i])
                / self.variances[y_data][i] ** 2
            )
            for i in range(2)
        ]
        return probabilities[1] - probabilities[0]


def _get_optimal_line(
    x_list: list[list[float]], y_list: list[int], zero_class: int
) -> np.ndarray:
    x_split: list[list[np.ndarray]] = [[], []]
    for idx, y_data in enumerate(y_list):
        cur_class = int(y_data != zero_class)
        x_split[cur_class].append(np.array(x_list[idx]))

    means: list[np.matrix] = []
    for x_data in x_split:
        means.append(np.mean(x_data, axis=0))

    scatter_within = np.sum(
        [
            np.sum(
                [
                    np.multiply(
                        np.matrix(x_split[c][i] - means[c]),
                        np.matrix(x_split[c][i] - means[c]).T,
                    )
                    for i in range(len(x_split[c]))
                ],
                axis=0,
            )
            for c in range(2)
        ],
        axis=0,
    )

    # normalize vector
    result = np.dot(np.linalg.inv(scatter_within), means[0] - means[1])
    return result / np.linalg.norm(result)
