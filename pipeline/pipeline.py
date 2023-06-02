import classifier


def run_pipeline(
    train_data: tuple[list[list[float]], list[int]],
    test_data: tuple[list[list[float]], list[int]],
    classifier_obj: classifier.BaseClassifier,
):
    """
    Trains the given classifier object on the training data, then returns its
    accuracy on the test data.
    """
    train_x, train_y = train_data
    classifier_obj.train(train_x, train_y)
    test_x, test_y = test_data
    pred_y = classifier_obj.predict(test_x)

    correct = 0
    for idx, pred in enumerate(pred_y):
        real = test_y[idx]
        if pred == real:
            correct += 1

    return correct / len(test_y)
