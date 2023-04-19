from typing import Union
import pandas as pd
import numpy as np
from math import sqrt
# import increments_rust_module as rm

simplex_table = pd.DataFrame(columns=['x1', 'x2', 'F'])     # Таблица


class ConstantsNamespace:
    DIMENSION: int = 2                                      # Размерность задачи
    EDGE_LENGTH: float = 0.75                                # Длина ребра симплекса
    ACCURACY: float = 0.1                                   # Точность
    START_POINT: tuple = (0.5, 2.0)                             # Начальные координаты


def target_function(x: tuple) -> float:
    """Целевая функция

    Args:
        x (tuple): _description_

    Returns:
        float: _description_
    """
    # return 2.8 * x[1]**2 + 1.9 * x[0] + 2.7 * x[0]**2 + 1.6 - 1.9 * x[1]        # Функция из методички
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2  # Функция Бута min: f(1, 3) = 0


def find_center(index: int, basis: tuple, n: int) -> tuple:
    """Функция для нахождения центра тяжести

        Args:
            index (int): _description_
            basis (tuple): _description_
            n (int): _description_
        Returns:
            tuple: _description_
    """
    center = (0, 0)
    for i in range(len(basis)):
        if i != index:
            center += basis[i]
    center /= n
    print(f'Центр тяжести: {center}')
    return center


def find_center_simplex(basis: tuple, n: int) -> tuple:
    """Функция нахождения векторного значения центра тяжести симплекса

        Args:
            basis (tuple]): _description_
            n (int): _description_
        Returns:
            tuple: _description_
    """
    center = sum(basis) / (n + 1)
    print(f'Векторное значение центра тяжести симплекса: {center}')
    return center


def find_increments(n: int, m: Union[float, int]) -> tuple:
    """Функция для нахождения инкрементов

        Args:
            n (int): _description_
            m (Union[float,int]): _description_
        Returns:
            tuple: _description_
    """
    b1 = ((sqrt(n + 1) - 1) / (n * sqrt(2))) * m
    b2 = ((sqrt(n + 1) + n - 1) / (n * sqrt(2))) * m
    print(f'Приращения: delta1 = {b1}, delta2 = {b2}.')
    return b1, b2


#
def find_max(basis: tuple) -> int:
    """Функция для нахождения ндекса макисмального значения функции в базизсном векторе

        Args:
            basis (tuple): _description_
        Returns:
            int: _description_
    """
    max_f = [target_function(basis[0]), 0]
    for i in range(0, len(basis)):  # 0 писать не обязательно, итерация и так с нуля
        if target_function(basis[i]) > max_f[0]:
            max_f = [target_function(basis[i]), i]
    print('Максимальное значение {0}'.format(max_f[0]))
    return max_f[1]


def find_min(basis: tuple) -> int:
    """Функция для нахождения индекса минимального значения функции в базизсном векторе

        Args:
            basis (tuple): _description_
        Returns:
            int: _description_
    """
    min_f = [target_function(basis[0]), 0]
    for i in range(0, len(basis)):
        if target_function(basis[i]) < min_f[0]:
            min_f = [target_function(basis[i]), i]
    print('Минимальное значение {0}'.format(min_f[0]))
    return min_f[1]


def reflected_vector(center: tuple, basis: tuple, index: int) -> tuple:
    """Функция для нахождения отраженного вектора

        Args:
            center (tuple): _description_
            basis (tuple): _description_
            index (int): _description_
        Returns:
            tuple: _description_
    """
    new_vector = 2 * np.array(center) - np.array(basis[index])
    print(f'Отраженный вектор: {new_vector}')
    return new_vector


def reduction(basis: list, index: int) -> tuple:
    """Редукция

    Args:
        basis (tuple): _description_
        index (int): _description_

    Returns:
        tuple: _description_
    """
    for i in range(len(basis)):
        if i != index:
            basis[i] = basis[index] + 0.5 * (basis[i] - basis[index])
            simplex_table.loc[len(simplex_table)] = np.append(basis[i], target_function(basis[i]))
    return tuple(basis)


def rebuild_basis(basis: tuple, new_vector: tuple, index: int) -> tuple:
    """Функция для изменения базиса

        Args:
            basis (tuple): _description_
            new_vector (tuple): _description_
            index (int): _description_

        Returns:
            tuple: _description_
        """
    new_basis = ()
    for i in range(len(basis)):
        if i != index:
            new_basis += (basis[i], )
        else:
            new_basis += (new_vector, )
    return new_basis


def check_end(center: float, basis: tuple, e: float) -> bool:
    """Функция проверки завершения алгоритма

    Args:
        center (float): _description_
        basis (tuple): _description_
        e (float): _description_

    Returns:
        flag (bool): _description_
    """
    print('\nПроверка')
    flag: bool = True
    for vector in basis:
        f_vector: float = target_function(vector)
        if f_vector - center > e:
            print(f'{f_vector}>{center}')
            flag = False
        else:
            print(f'{f_vector}<{center}')
    return flag


def show_basis(basis: tuple):
    """Функция для отображение текущего базиса

        Args:
            basis (tuple): _description_
        """
    for vector in basis:
        print(vector)


class SimplexMethod:
    """
    Класс для расчета симплекса
    """

    def __init__(self, n: int, m: Union[float, int], e: float, x: tuple):
        """Конструктор класса SimplexMethod

        Args:
            n (int): _description_
            m (int): _description_
            e (float): _description_
            x (tuple): _description_
        """
        self.n: int = n
        self.m: float = m
        self.e: float = e
        self.basis: tuple = ()
        self.increments: tuple = find_increments(n, m)
        self.__start_basis(x)

    def __start_basis(self, x: tuple):
        """Функция для инициализирования стартового базиса

        Args:
            x (tuple): Стартовая точка
        """
        start_basis = np.zeros((self.n+1, self.n))
        start_basis[0] = x
        start_basis[1][0], start_basis[1][1] = start_basis[0] + self.increments
        start_basis[2][1], start_basis[2][0] = start_basis[0] + self.increments
        self.basis = start_basis
        simplex_table.loc[0] = np.append(self.basis[0], target_function(self.basis[0]))
        simplex_table.loc[1] = np.append(self.basis[1], target_function(self.basis[1]))
        simplex_table.loc[2] = np.append(self.basis[2], target_function(self.basis[2]))

    def simplex_run(self, i: int) -> bool:
        """
        This is a function that's evaluate simplex table?.
        Args:
            i (int): Текущая итерация

        Returns:
            bool: Окончание метода
        """
        print(f'\nИтерация {i}:\n')
        print('Базис:')
        show_basis(self.basis)
        max_f: int = find_max(self.basis)
        center: tuple = find_center(max_f, self.basis, self.n)
        new: tuple = reflected_vector(center, self.basis, max_f)
        if target_function(new) >= target_function(self.basis[max_f]):
            print('Уменьшение целевой функции не наблюдается, проведем редукцию:')
            simplex_table.loc[len(simplex_table)] = ['-', '-', '-']
            min_f: int = find_min(self.basis)
            self.basis = reduction(list(self.basis).copy(), min_f)
        else:
            self.basis = rebuild_basis(self.basis, new, max_f)
            simplex_table.loc[len(simplex_table)] = np.append(self.basis[max_f], target_function(self.basis[max_f]))
        f_center = target_function(find_center_simplex(self.basis, self.n))
        print(f'Численное значение центра тяжести симплекса: {f_center}')

        if not check_end(f_center, self.basis, self.e):
            print('Условие не выполнено\n')
            print(simplex_table)
            return True
        print('Условие выполнено!\n')
        print(f'Приближенное решение: {self.basis[find_min(self.basis)]}')
        return False

    def simplex_method(self):
        i = 0
        while self.simplex_run(i):
            i += 1
        print('\n\nИтоговая таблица векторов: \n', simplex_table)


if __name__ == "__main__":
    constants = ConstantsNamespace()
    s1 = SimplexMethod(constants.DIMENSION, constants.EDGE_LENGTH, constants.ACCURACY, constants.START_POINT)
    s1.simplex_method()
