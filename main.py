"""
Main entry point for program.
"""
import classifier
import data_loading
import pipeline


def main():
    """
    Main function.
    """
    all_rows = data_loading.load_from_csv("dataset/iris.data")
    train_rows, test_rows = data_loading.stratified_train_test_split(
        all_rows,
        ratio_train=0.5,
    )
    formatter = data_loading.Formatter()
    acc = pipeline.run_pipeline(
        formatter.format_rows(train_rows),
        formatter.format_rows(test_rows),
        classifier.KNearestClassifier(k=7),
    )
    print(f"Accuracy: {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
