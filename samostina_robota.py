import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def scalar_product(row1, row2):
    return np.dot(row1, row2)

def find_min_scalar_product(matrix):
    m, n = matrix.shape
    min_product = np.inf
    min_indices = (1, 1)
    product_values = []

    def calculate_product(i, j):
        nonlocal min_product, min_indices
        product = scalar_product(matrix[i], matrix[j])
        if product < min_product:
            min_product = product
            min_indices = (i, j)
        product_values.append(product)

    Parallel(n_jobs=-1)(delayed(calculate_product)(i, j) for i in range(m) for j in range(i+1, m))

    return min_indices, product_values

# Пример использования
matrix = np.random.rand(5, 3)  # Замените размерность и данные матрицы на соответствующие вашим требованиям
result, product_values = find_min_scalar_product(matrix)
print("Минимальное скалярное произведение:", result)

# Построение графика
plt.plot([], [])  # Создаем пустой график без данных
plt.scatter(result[0], [1], color='blue', label=f'Координата {result[1]}')  # Отображение координаты
plt.xlabel('Номер пары рядов')
plt.ylabel('Скалярное произведение')
plt.title('Отображение выбранных координат')
plt.legend()
plt.show()