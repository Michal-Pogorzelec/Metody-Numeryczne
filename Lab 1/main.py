import math
import numpy as np
from typing import List, Tuple, Union


# Zadanie 1
def cylinder_area(r: float, h: float) -> float:
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if r < 0 or h < 0:
        return math.nan
    return (2 * math.pi * r * h) + (2 * math.pi * r**2)


# Zadanie 2
fst_seq = np.arange(1, 10, 0.4)
print(fst_seq)
sec_seq = np.linspace(1, 10)
print(sec_seq)


# Zadanie 3
def fib(n: int) -> Union[None, List[int], np.ndarray]:
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """

    if n <= 0 or type(n) is not int:
        return None
    result = [1, 1]
    if n == 1:
        return [1]
    for elem in range(0, n - 2):
        result += [result[-1] + result[-2]]

    return np.array([result])

# print(fib(15))


# Zadanie 4
def matrix_calculations(a: float) -> Tuple:
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """

    M = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    Mdet = np.linalg.det(M)
    if Mdet == 0:
        Minv = math.nan
    else:
        Minv = np.linalg.inv(M)
    Mt = np.transpose(M)

    return Minv, Mt, Mdet


# Zadanie 5
# [[3, 0, -2, 4], [1, 1, 1, 3], [-2, 1, 1, 0], [4, 5, 6, 1]]

M = np.array([[3, 0, -2, 4], [1, 1, 1, 3], [-2, 1, 1, 0], [4, 5, 6, 1]])
w1 = M[:, 2]
w2 = M[1, :]
print("Zadanie 5")
print("Macierz M:\n{0}".format(M))
print("Wektor w1:\n{0}".format(w1))
print("Wektor w2:\n{0}".format(w2))
print("Element [1,1]:\n{0}".format(M[0, 0]))
print("Element [3,3]:\n{0}".format(M[2, 2]))
print("Element [3,2]:\n{0}".format(M[2, 1]))
print("\n")


# Zadanie 6
def custom_matrix(m: int, n: int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.

    jeśli indeks wiersza jest większy od indeksu kolumny wartością komórki
    jest indeks wiersza, w przeciwnym wypadku wartością komórki jest indeks kolumny.
    """
    if type(m) == float or type(n) == float:
        return None
    if m <= 0 or n <= 0:
        return None
    result = np.zeros((m, n))
    for row_ind in range(m):
        for col_ind in range(n):
            if row_ind > col_ind:
                result[row_ind][col_ind] = row_ind
            else:
                result[row_ind][col_ind] = col_ind
    return result
print(custom_matrix(4, 6))

# Zadanie 7
v1 = np.array([1, 3, 13])
v2 = np.array([8, 5, -2])

result1 = np.multiply(4, v1)
# skorzystanie z własności broadcastingu
result2 = np.multiply(v2, -1) + 2
result3 = np.dot(v1, v2)
result4 = np.multiply(v1, v2)
print("Zadanie 7")
print(f"Result 1:\n{result1}")
print(f"Result 2:\n{result2}")
print(f"Result 3:\n{result3}")
print(f"Result 4:\n{result4}\n")



# Zadanie 8
M1 = np.array([[1, -7, 3], [-12, 3, 4], [5, 13, -3]])
result5 = np.multiply(M1, 3)
# skorzystanie z własności broadcastingu
result6 = M1 + 1
result7 = M1.transpose()
result8 = np.multiply(M1, v1)
result9 = np.multiply(v2.transpose(), M1)
print("Zadanie 8")
print(f"Result 5:\n{result5}")
print(f"Result 6:\n{result6}")
print(f"Result 7:\n{result7}")
print(f"Result 8:\n{result8}")
print(f"Result 9:\n{result9}")
