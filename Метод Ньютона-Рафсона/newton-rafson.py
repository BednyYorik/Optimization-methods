import pandas as pd
import numpy as np
from math import sqrt
from sympy import symbols, diff, sympify
from numpy.linalg import inv, cond

vector_table = pd.DataFrame(columns=['x1', 'x2', 'F'])


class ConstantNamespace:
    START_POINT: tuple = (-0.25, 0.25)
    DIMENSION: int = 2
    ACCURACY: float = 0.1
    # TARGET_FUNC: str = '2.8 * (x1 ** 2) + 1.9 * x0 + 2.7 * (x0 ** 2) + 1.6 - 1.9 * x1'  # Функиция из методички
    TARGET_FUNC: str = '(x0 + 2 * x1 - 7) ** 2 + (2 * x0 + x1 - 5) ** 2'  # Функция Бута min: f(1, 3) = 0


def target_function(n: int, x: tuple) -> float:
    """
    Целевая функция для оптимизации
    :param n: Размерность задачи
    :param x: Базисный вектор
    :return: Значение целевой функции
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


def direction(n: int) -> tuple:
    """
    Вычисление градиента целевой функции
    :param n: Размерность задачи
    :return: Вектор градиента целевой функции
    """
    x = symbols(f'x0:{n}', real=True)
    expr = sympify(ConstantNamespace.TARGET_FUNC, locals={'x{}'.format(i): x[i] for i in range(n)})
    dif = [diff(expr, sym) for sym in x]
    print(f'Градиент функции в произвольной точке: {dif}')
    return tuple(dif)


def direction_descent(gradient: tuple, hessian: tuple) -> tuple:
    """
    Вычисление направления спуска
    :param gradient: Градиент текущей итерации
    :param hessian:
    :return: Вектор направления спуска
    """
    p_new: tuple = (-1) * np.dot(hessian, gradient)
    return p_new


def step(gradient: tuple, hessian: tuple, p: tuple) -> float:
    """
    Вычисление шага
    :param gradient: Градиент целевой функции
    :param hessian: Матрица Гессе вычесленная для текущего базисного вектора
    :param p:
    :return: Шаг
    """
    h = np.inner(gradient, p)/np.dot(np.matmul(hessian, np.transpose(p)), p)
    print(f'Новое значение шага: {float(h)}')
    return float(h)


def check_end(gradient: tuple, e: float) -> bool:
    """
    Функция проверки окончания
    :param gradient: Градиент целевой функции
    :param e: Точность поиска
    :return: Метод завершен (inverted)
    """
    print('Проверка')
    acc = sqrt(sum(np.array(gradient) ** 2))
    print(f'Норма вектора градиента: {acc}')
    if acc < e:
        print(f'{acc}<={e}')
        return False
    print(f'{acc}>{e}')
    return True


class NewtonRafsonMethod:

    def __init__(self, n: int, e: float, x: tuple):
        """
        Конструктор класса FastestGradientDescent
        :param n: Размерность задачи
        :param e: Точность поиска
        :param x: Начальная точка
        """
        self.n: int = n
        self.e: float = e
        self.vector: tuple = x
        self.directions: tuple = direction(self.n)
        self.gradient: tuple = np.zeros(n)
        self.hessian: tuple = hessian_matrix(self.n)

    def nr_run(self, index: int):
        print(f'\n\nИтерация: {index}\n\n')
        print('Координаты базисного вектора: ', self.vector)
        grad = []
        for i in range(len(self.directions)):
            vector_x = {'x{}'.format(i): self.vector[i] for i in range(self.n)}
            grad.append(eval(str(self.directions[i]), vector_x))
        self.gradient = tuple(grad)
        print('Координаты вектора градиента: ', self.gradient)
        if not check_end(self.gradient, self.e):
            print('Условие выполнено!')
            print(f'Приближенное минимальное значение функции: {vector_table.loc[len(vector_table)-1][2]} '
                  f'в точке {self.vector}')
            return False
        print('Условие не выполнено. Численным методом определим значения шага:')
        calc_hess: tuple = calc_hessian(self.hessian, self.vector, self.n)
        if cond(calc_hess) <= 0:
            calc_hess = np.identity(self.n)
            p: tuple = direction_descent(self.gradient, inv(calc_hess))
        else:
            p: tuple = direction_descent(self.gradient, inv(calc_hess))
        h = step(self.gradient, calc_hess, p)
        self.vector = self.vector + h * np.dot(inv(calc_hess), self.gradient)
        vector_table.loc[len(vector_table)] = np.append(self.vector, target_function(self.n, self.vector))
        return True

    def newton_rafson(self):
        k: int = 0
        vector_table.loc[0] = np.append(self.vector, target_function(self.n, self.vector))
        while self.nr_run(k):
            k += 1
        print('\n\nФинальная таблица векторов:\n ', vector_table)


if __name__ == '__main__':
    constants = ConstantNamespace()
    s1 = NewtonRafsonMethod(constants.DIMENSION, constants.ACCURACY, constants.START_POINT)
    s1.newton_rafson()
