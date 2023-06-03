"""
Main entry point for program.
"""
import numpy as np

import classifier
import data_loading
import pipeline


def main():
    """
    Main function.
    """
    all_rows = data_loading.load_from_csv("dataset/iris.data")

    knn_acc = []
    fisher_acc = []
    for i in range(20):
        formatter = data_loading.Formatter()
        train_data, test_data = map(
            formatter.format_rows,
            data_loading.stratified_train_test_split(
                all_rows,
                ratio_train=0.5,
                seed=i,
            ),
        )
        knn_acc.append(
            pipeline.run_pipeline(
                train_data,
                test_data,
                classifier.KNearestClassifier(k=7),
            )
        )
        fisher_acc.append(
            pipeline.run_pipeline(
                train_data,
                test_data,
                classifier.FisherLDAClassifier(),
            )
        )

    print("KNN (K=7) Accuracy")
    print(f"Mean    : {np.mean(knn_acc):.6f}")
    print(f"Variance: {np.var(knn_acc):.6f}")

    print("Fisher's LDA Accuracy")
    print(f"Mean    : {np.mean(fisher_acc):.6f}")
    print(f"Variance: {np.var(fisher_acc):.6f}")


if __name__ == "__main__":
    main()
