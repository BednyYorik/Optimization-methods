import pandas as pd
import numpy as np
from math import *
from sympy import symbols, diff, sympify

vector_table = pd.DataFrame(columns=['x1', 'x2', 'F'])


class ConstantNamespace:

    START_POINT: tuple = (15, 15)                                                      # Начальная точка
    DIMENSION: int = 2                                                                      # Размерность задачи
    ACCURACY: float = 0.001                                                                   # Точность поиска
    # TARGET_FUNC: str = '2.8 * (x1 ** 2) + 1.9 * x0 + 2.7 * (x0 ** 2) + 1.6 - 1.9 * x1'      # Функиция из методички
    TARGET_FUNC: str = '(x0 + 2 * x1 - 7) ** 2 + (2 * x0 + x1 - 5) ** 2'  # Функция Бута min: f(1, 3) = 0
    # TARGET_FUNC: str = '((1.5 - x0 + (x0 * x1)) ** 2) + ((2.25 - x0 + (x0 * (x1 ** 2))) ** 2) ' \
    #                    '+ ((2.625 - x0 + (x0 * (x1 ** 3))) ** 2)'   # Функция Била min: f(3, 0.5) = 0


def target_function(n: int, x: tuple) -> float:
    """
    Целевая функция оптимизации
    :param int n: Размерность задачи
    :param tuple x: Координата базисного вектора
    :return: Значение функции для базисного вектора
    """
    return eval(ConstantNamespace.TARGET_FUNC, {'x{}'.format(i): x[i] for i in range(n)})


def hessian_matrix(n: int) -> tuple:
    """
    Вычисление матрицы Гессе
    :param n: Размерность задачи
    :return: Матрица Гессе для целевой функции
    """
    x = symbols(f'x0:{n}', real=True)
    hessian = []
    expr = sympify(ConstantNamespace.TARGET_FUNC, locals={f'x{i}': x[i] for i in range(n)})
    for i in range(n):
        for j in range(n):
            hessian.append(diff(expr, x[i], x[j]))
    hessian = np.reshape(np.array(hessian), (n, n))
    print(f'Матрица Гессе для целевой функции:\n {hessian}')
    return hessian


def direction(n: int) -> tuple:
    """
    Вычисление частных производных целевой функции
    :param n: Размерность задачи
    :return: Формулы вектора градиента
    """
    x = symbols(f'x0:{n}', real=True)
    expr = sympify(ConstantNamespace.TARGET_FUNC, locals={'x{}'.format(i): x[i] for i in range(n)})
    dif = [diff(expr, sym) for sym in x]
    print(f'Градиент функции в произвольной точке: {dif}')
    return tuple(dif)


def direction_descent(gradient: tuple, prev_gradient: tuple, p: tuple) -> tuple:
    """
    Вычисление направления спуска
    :param gradient: Градиент текущей итерации
    :param prev_gradient: Градиент предшедствующей итерации
    :param p: Предшедствующее направление спуска
    :return: Вектор направления спуска
    """
    betta = sum(np.array(gradient) ** 2)/sum(np.array(prev_gradient) ** 2)
    p_new = gradient + betta * np.array(p)
    return p_new


def step(p: tuple, gradient: tuple, hessian: tuple) -> float:
    """
    Вычисление шага
    :param p: Вектор направления спуска
    :param gradient: Градиент текущей итерации
    :param hessian: Матрица Гессе
    :return: Шаг
    """
    h = np.inner(gradient, p)/np.dot(np.matmul(np.matrix(hessian), np.array(p).T), np.array(p))
    print(f'Новое значение шага: {float(h)}')
    return float(h)


def check_end(gradient: tuple, eps: float) -> bool:
    """
    Проверка завершения метода
    :param gradient: Градиент текущей итерации
    :param eps: Точность поиска
    :return: Завершение операции (inverted)
    """
    print('Проверка')
    acc = sqrt(sum(np.array(gradient) ** 2))
    print(f'Норма вектора градиента: {acc}')
    if acc < eps:
        print(f'{acc}<={eps}')
        return False
    print(f'{acc}>{eps}')
    return True


def calc_hessian(hessian: tuple, vector: tuple, n: int) -> tuple:
    """
    Вычисление матрицы Гессе в точке
    :param hessian: Матрица Гессе
    :param vector: Базисный вектор (точка)
    :param n: Размерность задачи
    :return: Вычесленная матрица Гессе
    """
    vector_x: dict = {'x{}'.format(i): vector[i] for i in range(n)}
    calc_hess = np.reshape([eval(str(hessian[i][j]), vector_x) for i in range(n) for j in range(n)], (n, n))
    return tuple(calc_hess)


class FletcherRivesMethod:

    def __init__(self, n: int, eps: float, x: tuple):
        """
        Конструктор класса FletcherRivesMethod
        :param n: Размерность задачи
        :param eps: Точность поиска
        :param x: Начальная координата
        """
        self.n: int = n
        self.e: float = eps
        self.vector: tuple = x
        self.p: tuple = np.zeros(n)
        self.directions: tuple = direction(self.n)
        self.gradient: tuple = np.zeros(n)
        self.prev_gradient: tuple = np.zeros(n)
        self.hessian: tuple = hessian_matrix(self.n)

    def frm_run(self, index: int):
        print(f'\n\nИтерация: {index}\n\n')
        print('Координаты базисного вектора: ', self.vector)
        grad = []
        for i in range(len(self.directions)):
            vector_x: dict = {'x{}'.format(i): self.vector[i] for i in range(self.n)}
            grad.append(eval(str(self.directions[i]), vector_x))
        self.gradient = tuple(grad)
        print('Координаты вектора градиента: ', self.gradient)
        if index != 0:
            if not check_end(self.gradient, self.e):
                print('Условие выполнено!')
                print(f'Приближенное минимальное значение функции: {vector_table.loc[len(vector_table)-1][2]} '
                      f'в точке {self.vector}')
                return False
            else:
                print('Условие не выполнено. Определеем направление спуска:')
                self.p = direction_descent(self.gradient, self.prev_gradient, self.p)
        else:
            self.p = tuple((-1) * np.array(self.gradient))
        print(f'Направление спуска: {self.p}')
        calc_hess: tuple = calc_hessian(self.hessian, self.vector, self.n)
        h = step(self.p, self.gradient, calc_hess)
        self.prev_gradient = self.gradient
        self.vector = tuple(np.array(self.vector) - h * np.array(self.p))
        vector_table.loc[len(vector_table)] = np.append(self.vector, target_function(self.n, self.vector))
        return True

    def fletcher_rives_method(self):
        """
        Метод Флетчера-Ривса
        """
        k: int = 0
        vector_table.loc[0] = np.append(self.vector, target_function(self.n, self.vector))
        while self.frm_run(k):
            k += 1
        print('Финальная таблица векторов:\n ', vector_table)


if __name__ == '__main__':
    constants = ConstantNamespace()
    s1 = FletcherRivesMethod(constants.DIMENSION, constants.ACCURACY, constants.START_POINT)
    s1.fletcher_rives_method()
