import numpy as np


def transpose(M):
    Atr = np.array(M)
    for i in range(0, len(Atr)):
        for j in range(0, len(Atr)):
            Atr[i][j] = M[j][i]
    return Atr


def mul(A, B):
    n = len(A)
    C = np.array(A)
    for i in range(0, n):
        for j in rande(0, n):
            C[i][j] = 0
            for t in range(0, n):
                C[i][j] += A[i][t] * B[t][j]


def gauss(A, f):
    n = len(A)
    M = np.array(A)
    b = np.array(f)

    for k in range(0, n):
        for j in range(k + 1, n):
            num = float(M[j][k]) / M[k][k]
            for i in range(k, n):
                M[j][i] -= num * M[k][i]
            b[j] -= num * b[k]

    x = np.zeros(n)

    for k in range(n - 1, -1, -1):
        for i in range(k - 1, -1, -1):
            num = float(M[i][k]) / M[k][k]
            M[i][k] -= num * M[k][k]
            b[i] -= num * b[k]
    det = 1
    for i in range(0, n):
        x[i] = b[i] / M[i][i]
    for i in range(0, n):
        det *= M[i][i]

    # print("Determinant is: ", det)
    # print("Answer is: ", x)
    return x


def inverse(M):
    A = np.array(M)
    n = len(A)
    E = np.zeros((n, n))

    for i in range(0, n):
        E[i][i] = 1

    for k in range(0, n):
        for j in range(k + 1, n):
            d = float(A[j][k]) / A[k][k]
            for i in range(k, n):
                A[j][i] = A[j][i] - d * A[k][i]
            for i in range(0, n):
                E[j][i] = E[j][i] - d * E[k][i]

    for k in range(n - 1, -1, -1):
        koef = A[k][k]
        A[k][k] /= koef
        for i in range(0, n):
            E[k][i] /= koef
        for i in range(k - 1, -1, -1):
            koef = A[i][k] / A[k][k]
            for j in range(0, n):
                E[i][j] = E[i][j] - E[k][j] * koef
            A[i][k] = A[i][k] - A[k][k] * koef

    return E


def squareRootMethod(A, f, eps):
    n = len(A)
    M = np.array(A)
    b = np.array(f)
    Atr = transpose(M)
    C = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            for t in range(0, n):
                C[i][j] += Atr[i][t] * M[t][j]
    # перемножение матрицы A транспонированной и стобца b
    g = np.zeros(n)
    for i in range(0, n):
        for j in range(0, n):
            g[i] += b[j] * Atr[i][j]
    S = np.zeros((n, n))
    d = np.zeros(n)
    if C[0][0] > 0:
        d[0] = 1
    else:
        d[0] = -1
    for i in range(0, n):
        temp = 0
        for k in range(0, i):
            temp = temp + S[k][i] * S[k][i] * d[k]
        S[i][i] = np.sqrt(abs(C[i][i] - temp))
        d[i] = C[i][i] - temp
        if d[i] > 0:
            d[i] = 1
        else:
            d[i] = -1
        for j in range(i + 1, n):
            temp = 0
            for k in range(0, i):
                temp = temp + S[k][i] * S[k][j] * d[k]
            S[i][j] = (C[i][j] - temp) / (d[i] * S[i][i])

    # Выводим матрицу S:
    # print("Матрица S: \n", S)

    # Посчитаем матрицу DS:
    DS = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if S[i][j] != 0:
                DS[i][j] = S[i][j] * d[i]
            else:
                DS[i][j] = S[i][j]
    #  Найдем S ^ T
    Str = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            Str[i][j] = S[j][i]
    # Теперь посчитаем S ^ T * y = b
    x = np.zeros(n)
    y = np.zeros(n)
    cur = g[0] / Str[0][0]
    y[0] = cur
    for i in range(1, n):
        t = 0
        for j in range(0, i):
            cur = Str[i][j] * y[j]
            t = t + cur
        y[i] = (g[i] - t) / Str[i][i]
    for i in range(1, n):
        t = 0
        for j in range(0, i):
            cur = Str[i][j] * y[j]
            t = t + cur
        y[i] = (g[i] - t) / Str[i][i]

    # print()
    # print("Вывод y", y)
    # Решение DS * x = y
    for i in range(n - 1, -1, -1):
        t = 0
        for j in range(i + 1, n):
            cur = DS[i][j] * x[j]
            t = t + cur
        x[i] = (y[i] - t) / DS[i][i]

    print()
    # print("Вывод x:", x)
    return x


def simpleIterationMethod(A, f, eps):
    M = np.array(A)
    b = np.array(f)
    n = len(A)
    MT = np.array(transpose(M))  # создание транспортированной матрицы А
    for i in range(0, n):
        for j in range(0, n):
            MT[i][j] = M[j][i]

    tempM = np.array(M)  # результат перемножения матрицы А транспонированной на А
    for i in range(0, n):
        for j in range(0, n):
            temp = float(0)
            for k in range(0, n):
                temp += MT[i][k] * M[k][j]
            tempM[i][j] = temp

    norma = -1e10
    for i in range(0, n):
        temp = 0
        for j in range(0, n):
            temp += abs(tempM[i][j])
        norma = max(norma, temp)

    for i in range(0, n):
        for j in range(0, n):
            tempM[i][j] /= norma

    E = np.zeros((n, n))  # инициализация единичной матрицы
    for i in range(0, n):
        E[i][i] = 1

    for i in range(0, n):  # результат - первая матрица в формуле
        for j in range(0, n):
            E[i][j] -= tempM[i][j]

    tempf = np.zeros(n)  # инициализация результата перемножения А ^ T на f
    for i in range(0, n):
        temp = 0
        for k in range(0, n):
            temp += MT[i][k] * f[k]
        tempf[i] = temp
    for i in range(0, n):
        tempf[i] /= norma

    x_k = np.zeros(n)  # иницализациявектора x на k итерации
    x_l = np.zeros(n)  # иницализация вектора х на k + 1 итерации

    # первая матрица - E. второй вектор - x. третий вектор - tempf
    k = 0
    while True:
        stop = True
        k += 1
        temp = np.zeros(n)
        for i in range(0, n):
            temp_value = 0
            for j in range(0, n):
                temp_value += E[i][j] * x_k[j]
            temp[i] = temp_value
        for i in range(0, n):
            temp[i] += tempf[i]
        for i in range(0, n):
            stop = (stop and (abs(x_k[i] - temp[i]) < eps))
            if not stop:
                break
        if stop:
            x_l = temp
            break
        else:
            x_k = temp
    # print("Результат:")
    # print(x_l)
    # print("\nКоличество проведенных операций:")
    # print(k)
    return x_l


def gaussSeidelMethod(A, f, eps):
    n = len(A)
    M = np.array(A)
    b = np.array(f)
    x_l = np.zeros(n)  # x_i на итерации k + 1
    x_k = np.zeros(n)  # x_i на итерации k
    p = np.zeros(n)
    # транспонирование матриц
    Atr = np.array(transpose(A))
    # перемножение матриц M и Atr
    C = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                C[i][j] += Atr[i][k] * M[k][j]
    G = np.zeros(n)
    for i in range(0, n):
        for j in range(0, n):
            G[i] += Atr[i][j] * b[j]

    # имеем Cx = G
    k = 0
    while True:
        x_k = np.array(x_l)
        k += 1
        for i in range(0, n):
            temp = 0
            for j in range(0, i):
                temp += C[i][j] * x_l[j]
            for j in range(i + 1, n):
                temp += C[i][j] * x_k[j]
            x_l[i] = (G[i] - temp) / C[i][i]

        flag = True
        for i in range(0, n):
            flag = flag and (abs(x_l[i] - x_k[i]) < eps)
        if flag:
            break

    # print("Решение:")
    # print(x_l)
    # print()
    # print("Количество итераций:\n", k)
    return x_l
