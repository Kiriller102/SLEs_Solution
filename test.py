import SLEsolution as sleSol
import numpy as np

if __name__ == '__main__':
    k = 2
    a = 0.2 * k
    A = np.array([[8.3, 2.62 + a, 4.10, 1.90],
                  [3.92, 8.45, 7.78 - a, 2.46],
                  [3.77, 7.21 + a, 8.04, 2.28],
                  [2.21, 3.65 - a, 1.69, 6.99]])
    f = np.array([-10.65 + a, 12.21, 15.45 - a, -8.35])

    print(sleSol.Gauss(A, f))
    print()

    print(sleSol.GaussSeidelMethod(A, f, 1e-5))
    print()

    print(sleSol.squareRootMethod(A, f, 1e-5))
    print()

    print(sleSol.SimpleIterationMethod(A, f, 1e-5))

