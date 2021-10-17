import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import string
import random

# Zadanie 1
x = np.linspace(-1, 1)
y = x ** 3 - 3 * x
plt.plot(x, y, label="f(x)")
plt.xlabel("Os X")
plt.ylabel("f(x)")
plt.title("Wykres 1")
plt.legend()
plt.grid(True)
plt.show()

x = np.linspace(-5, 5)
y = x ** 3 - 3 * x
plt.plot(x, y, label="f(x)")
plt.xlabel("Os X")
plt.ylabel("f(x)")
plt.title("Wykres 2")
plt.grid(True)
plt.legend()
plt.show()

x = np.linspace(0, 5)
y = x ** 3 - 3 * x
plt.plot(x, y, label="f(x)")
plt.xlabel("Os X")
plt.ylabel("f(x)")
plt.title("Wykres 3")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Zadanie 2

x = np.linspace(-10, 10, 200)
y = x ** 3 - 3 * x
plt.plot(x, y)
plt.xlabel("Os X")
plt.ylabel("f(x)")
plt.xlim(-1, 1)
plt.ylim(-5, 5)
plt.axis()
plt.title("Wykres 1/2")
plt.grid(True)
plt.show()

x = np.linspace(-10, 10, 200)
y = x ** 3 - 3 * x
plt.plot(x, y)
plt.xlabel("Os X")
plt.ylabel("f(x)")
plt.xlim(-10, -1)
plt.ylim(-15, 15)
plt.axis()
plt.title("Wykres 2/2")
plt.grid(True)
plt.show()

x = np.linspace(-10, 10, 200)
y = x ** 3 - 3 * x
plt.plot(x, y)
plt.xlabel("Os X")
plt.ylabel("f(x)")
plt.xlim(1, 10)
plt.ylim(-5, 15)
plt.axis()
plt.title("Wykres 3/2")
plt.grid(True)
plt.tight_layout()
plt.show()

# Zadanie 3

m = 2.5
v = 60 / 3.6
Q_J = (m * v ** 2) / 2
# 1 cal = 4,184 J
Q_KCal = Q_J * 0.0002388459
print("Wynik w dżulach: %.2f [J], wynik w kcal: %.4f [kcal]." % (Q_J, Q_KCal))

v = np.linspace(200, 0, 400)
Q = (3 * v ** 2) / 2
plt.subplot(2, 1, 1)
plt.plot(v, Q)
plt.title("Wykres liniowy")
plt.xlabel("V [m/s]")
plt.ylabel("Q [J]")
plt.grid(True)

v = np.linspace(200, 0, 400)
Q = (3 * v ** 2) / 2
plt.subplot(2, 1, 2)
plt.semilogy(v, Q)
plt.title("Wykres logarytmiczny")
plt.xlim(200, 0)
plt.xlabel("V [m/s]")
plt.ylabel("Q [J]")
plt.grid(True)
plt.tight_layout()
plt.show()


# Zadanie 4

def compare_plot(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
                 xlabel: str, ylabel: str, title: str, label1: str, label2: str):
    """Funkcja służąca do porównywania dwóch wykresów typu plot. 
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    x1(np.ndarray): wektor wartości osi x dla pierwszego wykresu,
    y1(np.ndarray): wektor wartości osi y dla pierwszego wykresu,
    x2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    y2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    xlabel(str): opis osi x,
    ylabel(str): opis osi y,
    title(str): tytuł wykresu ,
    label1(str): nazwa serii z pierwszego wykresu,
    label2(str): nazwa serii z drugiego wykresu.

    Returns:
    matplotlib.pyplot.figure: wykres zbiorów (x1,y1), (x2,y2) zgody z opisem z zadania 4
    """

    if (x1.shape != y1.shape or min(x1.shape) == 0) or \
            (x2.shape != y2.shape or min(x2.shape) == 0):
        return None

    fig, ax = plt.subplots()
    ax.plot(x1, y1, "b", linewidth=4, label=label1)
    ax.plot(x2, y2, "r", linewidth=2, label=label2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(True)
    return fig


# Zadanie 5
x = np.linspace(-20, 20, 100)
y1 = x + 2
y2 = x ** 2 - 2 * np.sin(x) + 3
compare_plot(x, y1, x, y2, xlabel="X", ylabel="Y", title="Rozwiązanie graficzne", label1="f(x)",
             label2="g(x)")
plt.show()


# Zadanie 6

def parallel_plot(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
                  x1label: str, y1label: str, x2label: str, y2label: str, title: str, orientation: str):
    """Funkcja służąca do stworzenia dwóch wykresów typu plot w konwencji subplot wertykalnie lub choryzontalnie.
    Szczegółowy opis w zadaniu 6.
    
    Parameters:
    x1(np.ndarray): wektor wartości osi x dla pierwszego wykresu,
    y1(np.ndarray): wektor wartości osi y dla pierwszego wykresu,
    x2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    y2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    x1label(str): opis osi x dla pierwszego wykresu,
    y1label(str): opis osi y dla pierwszego wykresu,
    x2label(str): opis osi x dla drugiego wykresu,
    y2label(str): opis osi y dla drugiego wykresu,
    title(str): tytuł wykresu,
    orientation(str): parametr przyjmujący wartość '-' jeżeli subplot ma posiadać dwa wiersze albo '|' jeżeli ma posiadać dwie kolumny.

    
    Returns:
    matplotlib.pyplot.figure: wykres zbiorów (x1,y1), (x2,y2) zgodny z opisem z zadania 6
    """

    if (x1.shape != y1.shape or min(x1.shape) == 1) or \
            (x2.shape != y2.shape or min(x2.shape) == 1):
        return None

    if len(np.unique(x1)) != len(x1) or len(np.unique(x2)) != len(x2):
        return None

    if orientation == "-":
        fig, axs = plt.subplots(2)
    elif orientation == "|":
        fig, axs = plt.subplots(1, 2)
    else:
        return None

    fig.suptitle(title)
    ax1 = axs[0]
    ax2 = axs[1]

    ax1.plot(x1, y1)
    ax2.plot(x2, y2)

    ax1.set(xlabel=x1label, ylabel=y1label)
    ax2.set(xlabel=x2label, ylabel=y2label)
    return fig


# Zadanie 7
w1 = np.linspace(0, 1, 100)
w2 = np.linspace(0, 20, 100)
x1 = (np.e ** w1) * np.cos(w1)
y1 = (np.e ** w1) * np.sin(w1)
x2 = (np.e ** w2) * np.cos(w2)
y2 = (np.e * w2) * np.sin(w2)
parallel_plot(x1, y1, x2, y2, "x", "y", "x", "y", "logarytmiczne spirale", "|")
plt.show()


# Zadanie 8

def log_plot(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str,
             title: str, log_axis: str):
    """Funkcja służąca do tworzenia wykresów ze skalami logarytmicznymi. 
    Szczegółowy opis w zadaniu 8.
    
    Parameters:
    x(np.ndarray): wektor wartości osi x,
    y(np.ndarray): wektor wartości osi y,
    xlabel(str): opis osi x,
    ylabel(str): opis osi y,
    title(str): tytuł wykresu ,
    log_axis(str): wartość oznacza:
        - 'x' oznacza skale logarytmiczną na osi x,
        - 'y' oznacza skale logarytmiczną na osi y,
        - 'xy' oznacza skale logarytmiczną na obu osiach.
    
    Returns:
    matplotlib.pyplot.figure: wykres zbiorów (x,y) zgody z opisem z zadania 8
    """

    if x.shape != y.shape or min(x.shape) == 0:
        return None

    if log_axis == "x":
        fig = plt.semilogx(x, y)
    elif log_axis == "y":
        fig = plt.semilogy(x, y)
    else:
        fig = plt.plot(x, y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return fig


# Zadanie 9
v = np.linspace(200, 0, 400)
Q = (3 * v ** 2) / 2

log_plot(x=v, y=Q, xlabel="V [m/s]", ylabel="Q [J]", title="Wykres liniowy", log_axis="xy")
plt.show()
log_plot(x=v, y=Q, xlabel="V [m/s]", ylabel="Q [J]", title="Wykres X-logarytmiczny", log_axis="x")
plt.show()
log_plot(x=v, y=Q, xlabel="V [m/s]", ylabel="Q [J]", title="Wykres Y-logarytmiczny", log_axis="y")
plt.show()
