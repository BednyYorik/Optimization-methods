from math import sqrt
from typing import Union
import pandas as pd
import numpy as np

simplex_table = pd.DataFrame(columns=['x1', 'x2', 'F'])  # Таблица


class ConstantsNamespace:
    DIMENSION: int = 2  # Размерность задачи
    EDGE_LENGTH: float = 0.75  # Длина ребра симплекса
    BETA: float = 1.85  # Параметр растяжения
    GAMMA: float = 0.1  # Параметр сжатия
    ACCURACY: float = 0.001  # Точность
    START_POINT: tuple = (15, 15)  # Начальные координаты


def target_function(x: tuple) -> float:
    # return 2.8 * x[1] ** 2 + 1.9 * x[0] + 2.7 * x[0] ** 2 + 1.6 - 1.9 * x[1]  # Функция из методички
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2                # Функция Бута min: f(1, 3) = 0


def find_increments(n: int, m: float) -> tuple:
    """Функция для нахождения приращений

        Args:
            n (int): Размерность задачи оптимизации
            m (float): Длинна ребра многогранника
        Returns:
            tuple: Приращения
    """
    b1 = ((sqrt(n + 1) - 1) / (n * sqrt(2))) * m
    b2 = ((sqrt(n + 1) + n - 1) / (n * sqrt(2))) * m
    print(f'Приращения: delta1 = {b1}, delta2 = {b2}.')
    return b1, b2


def check_f(basis: list) -> tuple:
    max_f = find_max(basis)
    mid_f = find_mid(basis, max_f)
    min_f = find_min(basis)
    return max_f, mid_f, min_f


def find_mid(basis: list, max_index: int) -> int:
    max_f = [target_function(basis[0]), 0]
    for i in range(len(basis)):
        if target_function(basis[i]) > max_f[0] and i != max_index:
            max_f = [target_function(basis[i]), i]
    print('Следующее за максимальным значение {0}'.format(max_f[0]))
    return max_f[1]


# Функция для нахождения ндекса макисмального значения функции в базизсном векторе
def find_max(basis: list) -> int:
    max_f = [target_function(basis[0]), 0]
    for i in range(len(basis)):
        if target_function(basis[i]) > max_f[0]:
            max_f = [target_function(basis[i]), i]
    print('Максимальное значение {0}'.format(max_f[0]))
    return max_f[1]


# Функция для нахождения индекса минимального значения функции в базизсном векторе
def find_min(basis: list) -> int:
    min_f = [target_function(basis[0]), 0]
    for i in range(0, len(basis)):
        if target_function(basis[i]) < min_f[0]:
            min_f = [target_function(basis[i]), i]
    print('Минимальное значение {0}'.format(min_f[0]))
    return min_f[1]


def find_center(index: int, basis: list, n: int) -> tuple:
    """Функция для нахождения центра тяжести

        Args:
            index (int): _description_
            basis (Union[list, tuple]): _description_
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


def find_center_neldermead(basis: list, n: int) -> tuple:
    """Функция нахождения центра тяжести многогранника на данном шаге

        Args:
            basis (Union[list, tuple]): _description_
            n (int): _description_
        Returns:
            tuple: _description_
    """
    res = sum(basis) / (n + 1)
    return res


# Функция для нахождения отраженного вектора
def reflected_vector(center: tuple, basis: list, index: int) -> tuple:
    new_vector = 2 * np.array(center) - np.array(basis[index])
    print(f'Векторное значение отраженного вектора: {new_vector}')
    return new_vector


def stretching(center: tuple, b: float, basis: list) -> tuple:
    return np.array(center) + b * (np.array(basis) - np.array(center))


def compression(center: tuple, g: float, basis: list) -> tuple:
    return np.array(center) + g * (np.array(basis) - np.array(center))


def check_end(center: float, basis: list, n: int, e: float) -> bool:
    """Функция проверки завершения алгоритма

    Args:
        center (float): _description_
        basis (tuple): _description_
        n (int): _description_
        e (float): _description_

    Returns:
        flag (bool): _description_
    """
    print('\nПроверка')
    sig = 0
    for vector in basis:
        sig += (target_function(vector) - center)**2
    sig = sqrt(sig/(n+1))
    print(f"Сигма: {sig}")
    if sig < e:
        print(f'{sig}<{e}')
        return False
    print(f'{sig}>{e}')
    return True


def reduction(basis: list, index: int) -> list:  # ай ай ай, плохая идея передавать список в функцию
    """Редукция

    Args:
        basis (): _description_
        index (int): _description_

    Returns:
        _type_: _description_
    """
    for i in range(len(basis)):
        if i != index:
            basis[i] = basis[index] + 0.5 * (basis[i] - basis[index])
            simplex_table.loc[len(simplex_table)] = np.append(basis[i], target_function(basis[i]))

    return basis


class NelderMeadMethod:
    """
    Класс для расчета симплекса
    """

    def __init__(self, n: int, m: float, e: float, b: float, g: float, x: tuple):
        """Конструктор класса SimplexMethod

        Args:
            n (int): Размерность задачи оптимизации
            m (float): Длинна ребра многогранника
            e (float): Точность поиска
            b (float): Параметр растяжения
            g (float): Параметр сжатия
            x (tuple): Координата начальной точки
        """
        self.n: int = n
        self.m: float = m
        self.e: float = e
        self.b: float = b
        self.g: float = g
        self.basis: list = np.zeros((n + 1, n))
        self.increments: tuple = find_increments(n, m)
        self.__start_basis(x)

    def __start_basis(self, x: tuple):
        """Функция для инициализирования стартового базиса

        Args:
            x (list): Координата начальной точки
        """
        self.basis[0] = x
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.basis[i+1][j] = self.basis[0][j] + self.increments[0]
                else:
                    self.basis[i+1][j] = self.basis[0][j] + self.increments[1]
        for i in range(self.n+1):
            simplex_table.loc[len(simplex_table)] = np.append(self.basis[i], target_function(self.basis[i]))

    def simplex_run(self, i: int) -> bool:
        """
        This is a function that's evaluate simplex table?.
        Args:
            i (integer): Текущая итерация

        Returns:
            bool: Окончание метода
        """
        print(f'\nИтерация {i}:\n')
        print('Базис:\n', self.basis)
        f_max, f_mid, f_min = check_f(self.basis.copy())
        flag = False
        center: tuple = find_center(f_max, self.basis, self.n)
        new: tuple = reflected_vector(center, self.basis, f_max)
        simplex_table.loc[len(simplex_table)] = np.append(new, target_function(new))
        print(f"Численное значение отраженного вектора: {target_function(new)}")
        if target_function(new) < target_function(self.basis[f_max]):
            print('Наблюдается уменьшение целевой функции, операция отражения закончилась успешно.')
            self.basis[f_max] = new
            if target_function(self.basis[f_max]) < target_function(self.basis[f_min]):
                print("Выполним операцию растяжения")
                stretching_x = stretching(center, self.b, self.basis[f_max])
                if target_function(stretching_x) < target_function(self.basis[f_max]):
                    print("Операция растяжения закончилась успешно")
                    self.basis[f_max] = stretching_x
                    flag = True
        if (target_function(self.basis[f_mid]) < target_function(new) < target_function(self.basis[f_max])) \
                and not flag:
            print("Выполним операцию сжатия")
            compression_x = compression(center, self.g, self.basis[f_max])
            if target_function(compression_x) < target_function(self.basis[f_max]):
                print("Операция сжатия закончилась успешно")
                self.basis[f_max] = compression_x
                simplex_table.loc[len(simplex_table)] = np.append(self.basis[f_max], target_function(self.basis[f_max]))
                flag = True

        if not flag:
            print("Выполним редукцию")
            self.basis = reduction(self.basis, f_min)

        f_center = target_function(find_center_neldermead(self.basis, self.n))
        if check_end(f_center, self.basis.copy(), self.n, self.e):
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
    s1 = NelderMeadMethod(constants.DIMENSION, constants.EDGE_LENGTH, constants.ACCURACY,
                          constants.BETA, constants.GAMMA, constants.START_POINT)
    s1.simplex_method()
