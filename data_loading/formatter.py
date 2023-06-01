"""
Contains class for formatting rows of data into BaseClassifier format.
"""


class Formatter:
    """
    Formatter class with consistent Y encoding across format_*() calls.
    """

    def __init__(self) -> None:
        self.y_encoding: dict[str, int] = {}
        self.y_decoding: list[str] = []
        self.next_available_encoding = 0

    def encode_y(self, y_val: str) -> int:
        """
        Converts a string Y value to its integer representation.
        """
        if y_val not in self.y_encoding:
            self.y_encoding[y_val] = self.next_available_encoding
            self.y_decoding.append(y_val)
            self.next_available_encoding += 1
        return self.y_encoding[y_val]

    def decode_y(self, y_code: int) -> str:
        """
        Converts the integer representation of a Y value to its real value.
        """
        return self.y_decoding[y_code]

    def format_row(self, row: list[str]) -> tuple[list[float], int]:
        """
        Splits the given rows of data into separate X and Y parts.

        Assumes the last column of data is the Y.
        """
        return [float(x) for x in row[:-1]], self.encode_y(row[-1])

    def format_rows(
        self, rows: list[list[str]]
    ) -> tuple[list[list[float]], list[int]]:
        """
        Splits the given rows of data into separate X and Y parts.

        Assumes the last column of data is the Y.
        """
        x_val: list[list[float]] = []
        y_val: list[int] = []
        for row in rows:
            row_x, row_y = self.format_row(row)
            x_val.append(row_x)
            y_val.append(row_y)
        return x_val, y_val
