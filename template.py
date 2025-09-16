class Matrix:
    def __init__(self, a: list[list[float]]):
        # напишите свою реализацию
        pass

    def __getitem__(self, index):
        # напишите свою реализацию
        pass

    def __setitem__(self, index, value):
        # напишите свою реализацию
        pass

    @property
    def T(self) -> "Matrix":
        # напишите свою реализацию
        pass

    @classmethod
    def eye(cls, size: int) -> "Matrix":
        # напишите свою реализацию
        pass

    @classmethod
    def create(cls, n: int, m: int, elem: float = 0) -> "Matrix":
        # напишите свою реализацию
        pass

    def __repr__(self) -> str:
        # напишите свою реализацию
        pass

    def __str__(self) -> str:
        # напишите свою реализацию
        pass

    def __add__(self, other: "Matrix") -> "Matrix":
        # напишите свою реализацию
        pass

    def __sub__(self, other: "Matrix") -> "Matrix":
        # напишите свою реализацию
        pass

    def __matmul__(self, other) -> "Matrix":
        # напишите свою реализацию
        pass

    def __mul__(self, other) -> "Matrix":
        # напишите свою реализацию
        pass

    def __rmul__(self, other: float) -> "Matrix":
        # напишите свою реализацию
        pass

    def __eq__(self, other: "Matrix") -> bool:
        # напишите свою реализацию
        pass

    def __ne__(self, other: "Matrix") -> bool:
        # напишите свою реализацию
        pass

    def tr(self) -> "Matrix":
        # напишите свою реализацию
        pass

    def det(self) -> float:
        # напишите свою реализацию
        pass

    def _get_minor(self, row: float, col: float) -> "Matrix":
        # напишите свою реализацию
        pass

    def is_sq(self) -> bool:
        # напишите свою реализацию
        pass

    def is_sym(self) -> bool:
        # напишите свою реализацию
        pass

    def norm(self, ord: str = 'fro') -> float:
        # напишите свою реализацию
        pass

    def __truediv__(self, scalar: float) -> "Matrix":
        # напишите свою реализацию
        pass


    def copy(self) -> "Matrix":
        # напишите свою реализацию
        pass

    def __pow__(self, power: int) -> "Matrix":
        # напишите свою реализацию
        pass

    def row_mult(self, i: int, scalar: float) -> None:
        # напишите свою реализацию
        pass

    def col_mult(self, j: int, scalar: float) -> None:
        # напишите свою реализацию
        pass

    def row_add(self, i: int, j: int, scalar: float = 1) -> None:
        # напишите свою реализацию
        pass

    def col_add(self, i: int, j: int, scalar: float = 1) -> None:
        # напишите свою реализацию
        pass

    def swap_rows(self, i: int, j: int) -> None:
        # напишите свою реализацию
        pass

    def swap_cols(self, i: int, j: int) -> None:
        # напишите свою реализацию
        pass

    @property
    def inv(self) -> "Matrix":
        # напишите свою реализацию
        pass


if __name__ == "__main__":
    # напишите тестовый код при необходимости
    pass