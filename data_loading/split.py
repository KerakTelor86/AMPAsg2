"""
Contains function for splitting data into train and test splits.
"""
import random


def stratified_train_test_split(
    data: list[list[str]],
    ratio_train: float,
    seed: int = 420691337,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Splits the given data by the y field (last column),
    stratified across all values of Y.

    Uses a fixed seed by default.
    """
    random.seed(seed)

    data_by_y: dict[str, list[list[str]]] = {}
    for datum in data:
        y_key = datum[-1]
        if y_key not in data_by_y:
            data_by_y[y_key] = []
        data_by_y[y_key].append(datum)

    train: list[list[str]] = []
    test: list[list[str]] = []
    for rows in data_by_y.values():
        random.shuffle(rows)

        size_train = round(len(rows) * ratio_train)
        train.extend(rows[:size_train])
        test.extend(rows[size_train:])

    return train, test
