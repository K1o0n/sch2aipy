class Matrix:
    def __init__(self,a: list[list[float]]):
        self.n = len(a)
        self.m = len(a[0])
        for i in range(self.n):
            if len(a[i]) != self.m:
                raise ValueError(f"\u001b[31;1mМатрица не прямоугольная\u001b[0m")
        self.mat = tuple(tuple(a[i][j] for j in range(self.m)) for i in range(self.n))
    def __getitem__(self, index):
        """Позволяет использовать:
        - Matrix[i][j] - элемент
        - Matrix[i] - строка
        - Matrix[i, j] - элемент
        - Matrix[:, j] - столбец
        - Matrix[i, :] - строка
        - Matrix[:, :] - вся матрица
        """
        if isinstance(index, (int, float)):
            # Matrix[i] - возвращаем i-ю строку
            if index < 0 or index >= self.n:
                raise IndexError(f"\u001b[31;1mИндекс строки {index} вне диапазона [0, {self.n-1}]\u001b[0m")
            return list(self.mat[index])  # Возвращаем список для возможности изменения
        
        elif isinstance(index, tuple) and len(index) == 2:
            i, j = index
            
            # Matrix[:, j] - возвращаем j-й столбец
            if i == slice(None) and isinstance(j, (int, float)):
                if j < 0 or j >= self.m:
                    raise IndexError(f"\u001b[31;1mИндекс столбца {j} вне диапазона [0, {self.m-1}]\u001b[0m")
                return [self.mat[row][j] for row in range(self.n)]  # Возвращаем список
            
            # Matrix[i, :] - возвращаем i-ю строку
            elif isinstance(i, (int, float)) and j == slice(None):
                if i < 0 or i >= self.n:
                    raise IndexError(f"\u001b[31;1mИндекс строки {i} вне диапазона [0, {self.n-1}]\u001b[0m")
                return list(self.mat[i])  # Возвращаем список
            
            # Matrix[:, :] - возвращаем всю матрицу
            elif i == slice(None) and j == slice(None):
                return [list(row) for row in self.mat]  # Возвращаем список списков
            
            # Matrix[i, j] - возвращаем конкретный элемент
            elif isinstance(i, (int, float)) and isinstance(j, (int, float)):
                if i < 0 or i >= self.n or j < 0 or j >= self.m:
                    raise IndexError(f"\u001b[31;1mИндекс ({i}, {j}) вне диапазона\u001b[0m")
                return self.mat[i][j]
            
            else:
                raise TypeError("\u001b[31;1mНеверный формат индекса\u001b[0m")
        
        else:
            raise TypeError("\u001b[31;1mНеверный тип индекса\u001b[0m")
    def __setitem__(self, index, value):
        """Позволяет использовать:
        - Matrix[i, j] = value - установка элемента
        - Matrix[:, j] = value - установка всего столбца
        - Matrix[i, :] = value - установка всей строки
        """
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            
            # Matrix[:, j] = value - установка всего столбца
            if i == slice(None) and isinstance(j, (int, float)):
                if j < 0 or j >= self.m:
                    raise IndexError(f"\u001b[31;1mИндекс столбца {j} вне диапазона [0, {self.m-1}]\u001b[0m")
                if not hasattr(value, '__iter__') or len(value) != self.n:
                    raise ValueError(f"\u001b[31;1mДля столбца нужно передать {self.n} значений\u001b[0m")
                
                new_mat = list(list(row) for row in self.mat)
                for row in range(self.n):
                    new_mat[row][j] = value[row]
                self.mat = tuple(tuple(row) for row in new_mat)
            
            # Matrix[i, :] = value - установка всей строки
            elif isinstance(i, (int, float)) and j == slice(None):
                if i < 0 or i >= self.n:
                    raise IndexError(f"\u001b[31;1mИндекс строки {i} вне диапазона [0, {self.n-1}]\u001b[0m")
                if not hasattr(value, '__iter__') or len(value) != self.m:
                    raise ValueError(f"\u001b[31;1mДля строки нужно передать {self.m} значений\u001b[0m")
                
                new_mat = list(list(row) for row in self.mat)
                new_mat[i] = list(value)
                self.mat = tuple(tuple(row) for row in new_mat)
            
            # Matrix[i, j] = value - установка элемента
            elif isinstance(i, (int, float)) and isinstance(j, (int, float)):
                if i < 0 or i >= self.n or j < 0 or j >= self.m:
                    raise IndexError(f"\u001b[31;1mИндекс ({i}, {j}) вне диапазона\u001b[0m")
                
                new_mat = list(list(row) for row in self.mat)
                new_mat[i][j] = value
                self.mat = tuple(tuple(row) for row in new_mat)
            
            else:
                raise TypeError("\u001b[31;1mНеверный формат индекса для установки значения\u001b[0m")
        
        else:
            raise TypeError("\u001b[31;1mДля установки значения используйте Matrix[i, j] = value\u001b[0m")
    @property
    def T(self) -> "Matrix":
        """Транспонированная матрица (свойство, как в NumPy)"""
        result = [[self.mat[j][i] for j in range(self.n)] for i in range(self.m)]
        return Matrix(result)
    @classmethod
    def eye(cls, size: int) -> "Matrix":
        """Создает единичную матрицу определенного размера

        Args:
            size (int): Размер единичной матрицы

        Returns:
            Matrix: Единичная матрица
        """
        _matrix = Matrix([[1 if i == j else 0 for j in range(size)] for i in range(size)])
        return _matrix
    @classmethod
    def create(cls, n: int, m: int, elem: float=0) -> "Matrix":
        """Создает матрицу заполненную определенным значением

        Args:
            n (int): Размер внешний
            m (int): Размер внутренний
            elem (float): Значение для заполнения матрицы

        Returns:
            Matrix: Матрицы заполненная 
        """
        _matrix = Matrix([[elem for _ in range(m)] for _ in range(n)])
        return _matrix
    def __repr__(self) -> str:
        _ret = "(\n  " + ",\n  ".join(str(tuple(round(self.mat[i][j], 6) for j in range(self.m))) for i in range(self.n)) + "\n)" 
        return _ret
    def __str__(self) -> str:
        _ret = "(\n  " + ",\n  ".join(str(tuple(round(self.mat[i][j], 6) for j in range(self.m))) for i in range(self.n)) + "\n)" 
        return _ret
    def __add__(self: "Matrix", other: "Matrix") -> "Matrix":
        if self.n != other.n or self.m != other.m:
            raise ValueError("\u001b[31;1mРазмеры матриц не совпадают\u001b[0m")
        result = [
            [self.mat[i][j] + other.mat[i][j] for j in range(self.m)]
            for i in range(self.n)
        ]
        return Matrix(result)
    def __sub__(self, other) -> "Matrix":
        if self.n != other.n or self.m != other.m:
            raise ValueError("\u001b[31;1mРазмеры матриц не совпадают\u001b[0m")
        
        result = [
            [self.mat[i][j] - other.mat[i][j] for j in range(self.m)]
            for i in range(self.n)
        ]
        return Matrix(result)
    def __matmul__(self, other) -> "Matrix":
        if isinstance(other, Matrix):  # Умножение матриц
            if self.m != other.n:
                raise ValueError("\u001b[31;1mНесовместимые размеры матриц для умножения\u001b[0m")
            
            result = [[0 for _ in range(other.m)] for _ in range(self.n)]
            for i in range(self.n):
                for j in range(other.m):
                    for k in range(self.m):
                        result[i][j] += self.mat[i][k] * other.mat[k][j]
            return Matrix(result)
        else:
            raise ValueError("\u001b[31;1mОбоими параметрами должны быть матрицы\u001b[0m")
    def __mul__(self, other) -> "Matrix":
        if isinstance(other, (int, float)):  # Умножение на скаляр
            result = [[self.mat[i][j] * other for j in range(self.m)] for i in range(self.n)]
            return Matrix(result)
        else:
            raise ValueError("\u001b[31;1mНедопустимый тип для умножения с матрицей\u001b[0m")
    def __rmul__(self, other: float) -> "Matrix":
        """Умножение скаляра на матрицу (other * self)"""
        return self * other
    def __eq__(self, other: "Matrix") -> bool:
        """Проверка на равенство матриц"""
        if self.n != other.n or self.m != other.m:
            return False
        return all(self.mat[i][j] == other.mat[i][j] 
                  for i in range(self.n) for j in range(self.m))
    def __ne__(self, other: "Matrix") -> bool:
        """Проверка на неравенство матриц"""
        return not self.__eq__(other)
    def tr(self) -> "Matrix":
        """Транспонирование матрицы"""
        result = [[self.mat[j][i] for j in range(self.n)] for i in range(self.m)]
        return Matrix(result)
    def det(self) -> float:
        """Вычисление определителя матрицы (только для квадратных матриц)"""
        if self.n != self.m:
            raise ValueError("\u001b[31;1mОпределитель можно вычислить только для квадратной матрицы\u001b[0m")
        
        if self.n == 1:
            return self.mat[0][0]
        elif self.n == 2:
            return self.mat[0][0] * self.mat[1][1] - self.mat[0][1] * self.mat[1][0]
        else:
            det = 0
            for j in range(self.m):
                minor = self._get_minor(0, j)
                sign = 1 if j % 2 == 0 else -1
                det += sign * self.mat[0][j] * minor.det()
            return det
    def _get_minor(self, row: float, col: float) -> "Matrix":
        """Получение минора матрицы"""
        minor = []
        for i in range(self.n):
            if i == row:
                continue
            minor_row = []
            for j in range(self.m):
                if j == col:
                    continue
                minor_row.append(self.mat[i][j])
            minor.append(minor_row)
        return Matrix(minor)
    def trace(self) -> float:
        """След матрицы (сумма элементов главной диагонали)"""
        if self.n != self.m:
            raise ValueError("\u001b[31;1mСлед можно вычислить только для квадратной матрицы\u001b[0m")
        return sum(self.mat[i][i] for i in range(self.n))
    def is_sq(self) -> bool:
        """Проверка, является ли матрица квадратной"""
        return self.n == self.m
    def is_sym(self) -> bool:
        """Проверка, является ли матрица симметричной"""
        if not self.is_sq():
            return False
        return self == self.T
    def norm(self, ord: str = 'fro') -> float:
        """Норма матрицы"""
        if ord == 'fro':  # Норма Фробениуса
            return sum(self.mat[i][j] ** 2 for i in range(self.n) for j in range(self.m)) ** 0.5
        elif ord == 'inf':  # Бесконечная норма
            return max(sum(abs(self.mat[i][j]) for j in range(self.m)) for i in range(self.n))
        else:
            raise ValueError("\u001b[31;1mНеизвестный тип нормы\u001b[0m")
    def __truediv__(self, scalar: float) -> "Matrix":
        """Деление матрицы на скаляр"""
        if scalar == 0:
            raise ZeroDivisionError("\u001b[31;1mДеление на ноль\u001b[0m")
        return self * (1 / scalar)
    def __floordiv__(self, scalar: float) -> "Matrix":
        """Деление матрицы на скаляр"""
        if scalar == 0:
            raise ZeroDivisionError("\u001b[31;1mДеление на ноль\u001b[0m")
        w = Matrix.create(self.n, self.m)
        for i in range(self.n):
            for j in range(self.m):
                w.mat[i][j] = self.mat[i][j] // scalar
        return w
    def copy(self) -> "Matrix":
        """Создание копии матрицы"""
        return Matrix([list(row) for row in self.mat])
    def __pow__(self, power: int) -> "Matrix":
        """Возведение матрицы в степень (только для целых неотрицательных степеней). Ассимптотика `O(n^3 log k)`

        Args:
            power (int): Степень в которую нужно возвести матрицу

        Returns:
            Matrix: Матрица в степени power
        """
        if not isinstance(power, int) or power < 0:
            raise ValueError("\u001b[31;1mСтепень должна быть целым неотрицательным числом\u001b[0m")
        if not self.is_sq():
            raise ValueError("\u001b[31;1mВозведение в степень возможно только для квадратных матриц\u001b[0m")
        result = Matrix.eye(self.n)
        while power:
            if power & 1:
                result = result @ self
            self = self @ self
            power >>= 1
        
        return result
    def __iadd__(self, other: "Matrix") -> "Matrix":
        """Операция +="""
        if self.n != other.n or self.m != other.m:
            raise ValueError("\u001b[31;1mРазмеры матриц не совпадают\u001b[0m")
        new_mat = list(list(row) for row in self.mat)
        for i in range(self.n):
            for j in range(self.m):
                new_mat[i][j] += other.mat[i][j]
        self.mat = tuple(tuple(row) for row in new_mat)
        return self
    def __isub__(self, other: "Matrix") -> "Matrix":
        """Операция -="""
        if self.n != other.n or self.m != other.m:
            raise ValueError("\u001b[31;1mРазмеры матриц не совпадают\u001b[0m")
        new_mat = list(list(row) for row in self.mat)
        for i in range(self.n):
            for j in range(self.m):
                new_mat[i][j] -= other.mat[i][j]
        self.mat = tuple(tuple(row) for row in new_mat)
        return self
    def __imul__(self, other) -> "Matrix":
        """Операция *= (умножение на скаляр, строки или столбца)"""
        if isinstance(other, (int, float)):
            # Умножение всей матрицы на скаляр
            new_mat = list(list(row) for row in self.mat)
            for i in range(self.n):
                for j in range(self.m):
                    new_mat[i][j] *= other
            self.mat = tuple(tuple(row) for row in new_mat)
            return self
        raise ValueError("\u001b[31;1mНедопустимый тип для умножения с матрицей\u001b[0m")
    def __itruediv__(self, other) -> "Matrix":
        """Операция /= (деление на скаляр, строки или столбца)"""
        if isinstance(other, (int, float)):
            # Деление всей матрицы на скаляр
            if other == 0:
                raise ZeroDivisionError("\u001b[31;1mДеление на ноль\u001b[0m")
            new_mat = list(list(row) for row in self.mat)
            for i in range(self.n):
                for j in range(self.m):
                    new_mat[i][j] /= other
            self.mat = tuple(tuple(row) for row in new_mat)
            return self
            
        raise ValueError("\u001b[31;1mНедопустимый тип для деления с матрицей\u001b[0m")
    def __ifloordiv__(self, other) -> "Matrix":
        """Операция //= (целочисленное деление на скаляр, строки или столбца)"""
        if isinstance(other, (int, float)):
            # Деление всей матрицы на скаляр
            if other == 0:
                raise ZeroDivisionError("\u001b[31;1mДеление на ноль\u001b[0m")
            new_mat = list(list(row) for row in self.mat)
            for i in range(self.n):
                for j in range(self.m):
                    new_mat[i][j] //= other
            self.mat = tuple(tuple(row) for row in new_mat)
            return self
            
        raise ValueError("\u001b[31;1mНедопустимый тип для деления с матрицей\u001b[0m")
    def tolist(self) -> list[list[float]]:
        """Преобразование матрицы в список списков"""
        return [list(row) for row in self.mat]
    def shape(self) -> tuple[int, int]:
        """Возвращает размер матрицы в виде кортежа (n, m)"""
        return (self.n, self.m)
    def size(self) -> int:
        """Возвращает общее количество элементов в матрице"""
        return self.n * self.m
    def fill(self, value: float) -> None:
        """Заполняет всю матрицу заданным значением"""
        new_mat = [[value for _ in range(self.m)] for _ in range(self.n)]
        self.mat = tuple(tuple(row) for row in new_mat)
    def clear(self) -> None:
        """Очищает матрицу, заполняя её нулями"""
        self.fill(0)
    def row_mult(self, i: int, scalar: float) -> None:
        """Умножение строки i на скаляр"""
        if i < 0 or i >= self.n:
            raise IndexError(f"\u001b[31;1mИндекс строки {i} вне диапазона [0, {self.n-1}]\u001b[0m")
        new_mat = list(list(row) for row in self.mat)
        for j in range(self.m):
            new_mat[i][j] *= scalar
        self.mat = tuple(tuple(row) for row in new_mat)
    def col_mult(self, j: int, scalar: float) -> None:
        """Умножение столбца j на скаляр"""
        if j < 0 or j >= self.m:
            raise IndexError(f"\u001b[31;1mИндекс столбца {j} вне диапазона [0, {self.m-1}]\u001b[0m")
        new_mat = list(list(row) for row in self.mat)
        for i in range(self.n):
            new_mat[i][j] *= scalar
        self.mat = tuple(tuple(row) for row in new_mat)
    def __radd__(self, other: "Matrix") -> "Matrix":
        """Операция сложения с другой матрицей (other + self)"""
        return self + other
    def __rsub__(self, other: "Matrix") -> "Matrix":
        """Операция вычитания с другой матрицей (other - self)"""
        return other - self
    def __rmul__(self, other: float) -> "Matrix":
        """Операция умножения с числом"""
        return self * other
    def row_add(self, i: int, j: int, scalar: float = 1) -> None:
        """Прибавляет к строке i строку j, умноженную на скаляр"""
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            raise IndexError(f"\u001b[31;1mИндексы строк {i} или {j} вне диапазона [0, {self.n-1}]\u001b[0m")
        new_mat = list(list(row) for row in self.mat)
        for col in range(self.m):
            new_mat[i][col] += new_mat[j][col] * scalar
        self.mat = tuple(tuple(row) for row in new_mat)
    def col_add(self, i: int, j: int, scalar: float = 1) -> None:
        """Прибавляет к столбцу i столбец j, умноженный на скаляр"""
        if i < 0 or i >= self.m or j < 0 or j >= self.m:
            raise IndexError(f"\u001b[31;1mИндексы столбцов {i} или {j} вне диапазона [0, {self.m-1}]\u001b[0m")
        new_mat = list(list(row) for row in self.mat)
        for row in range(self.n):
            new_mat[row][i] += new_mat[row][j] * scalar
        self.mat = tuple(tuple(row) for row in new_mat)
    def swap_rows(self, i: int, j: int) -> None:
        """Меняет местами строки i и j"""
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            raise IndexError(f"\u001b[31;1mИндексы строк {i} или {j} вне диапазона [0, {self.n-1}]\u001b[0m")
        new_mat = list(list(row) for row in self.mat)
        new_mat[i], new_mat[j] = new_mat[j], new_mat[i]
        self.mat = tuple(tuple(row) for row in new_mat)
    def swap_cols(self, i: int, j: int) -> None:
        """Меняет местами столбцы i и j"""
        if i < 0 or i >= self.m or j < 0 or j >= self.m:
            raise IndexError(f"\u001b[31;1mИндексы столбцов {i} или {j} вне диапазона [0, {self.m-1}]\u001b[0m")
        new_mat = list(list(row) for row in self.mat)
        for row in range(self.n):
            new_mat[row][i], new_mat[row][j] = new_mat[row][j], new_mat[row][i]
        self.mat = tuple(tuple(row) for row in new_mat)
    def __bool__(self) -> bool:
        """Проверка, является ли матрица нулевой (все элементы равны нулю)"""
        return any(self.mat[i][j] != 0 for i in range(self.n) for j in range(self.m))
    @property
    def rows(self):
        """
        Итератор по строкам.
        Каждый элемент — это копия строки в виде списка.
        """
        for row in self.mat:
            yield list(row)
    @property
    def cols(self):
        """
        Итератор по столбцам.
        Каждый элемент — это список значений соответствующего столбца.
        """
        for j in range(self.m):
            yield [self.mat[i][j] for i in range(self.n)]
    @property
    def inv(self) -> "Matrix":
        """Вычисление обратной матрицы (только для квадратных матриц)"""
        if not self.is_sq():
            raise ValueError("\u001b[31;1mОбратную матрицу можно вычислить только для квадратной матрицы\u001b[0m")
        
        n = self.n
        aug = [list(self.mat[i]) + [1 if i == j else 0 for j in range(n)] for i in range(n)]
        
        for i in range(n):
            pivot = aug[i][i]
            if pivot == 0:
                for r in range(i + 1, n):
                    if aug[r][i] != 0:
                        aug[i], aug[r] = aug[r], aug[i]
                        pivot = aug[i][i]
                        break
            if pivot == 0:
                raise ValueError("\u001b[31;1mМатрица вырождена и не имеет обратной\u001b[0m")
            
            for j in range(2 * n):
                aug[i][j] /= pivot
            
            for r in range(n):
                if r != i:
                    factor = aug[r][i]
                    for j in range(2 * n):
                        aug[r][j] -= factor * aug[i][j]
        
        inv_mat = [row[n:] for row in aug]
        return Matrix(inv_mat)
if __name__ == "__main__":
    e = Matrix([[3,2,3], [4,5,6], [7,8,9]])
    print(e)
    print(e.det())
