"""
Main entry point for program.
"""
import data_loading


def main():
    """
    Main function.
    """
    all_rows = data_loading.load_from_csv("dataset/iris.data")
    train_rows, test_rows = data_loading.stratified_train_test_split(
        all_rows,
        ratio_train=0.5,
    )
    print(len(train_rows), len(test_rows))


if __name__ == "__main__":
    main()
