"""
Contains function for loading data from CSV.
"""
import csv


def load_from_csv(
    csv_path: str,
    delimiter: str = ",",
) -> list[list[str]]:
    """
    Loads data from CSV.
    """
    rows = []
    with open(csv_path, encoding="utf-8") as csv_file:
        reader = csv.reader(
            csv_file,
            delimiter=delimiter,
        )
        for row in reader:
            if len(row) > 0:
                rows.append(row)
    return rows
