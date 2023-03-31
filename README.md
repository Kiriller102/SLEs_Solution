## Introduction

This project implements methods for solving systems of linear equations (SLEs): Gaussian elimination, square root method, simple iteration method, and Gauss-Seidel iteration. Each of these methods solves an SLE using algorithms that compute the values of the unknown variables in the equation.
## Installation

Before using this project, you need to install Python on your computer. You can download the latest version of Python from the official website: https://www.python.org/downloads/

To use this project, you will also need to install the NumPy library, which provides support for numerical computing with Python. You can install NumPy using pip, the Python package manager, by running the following command in your terminal or command prompt:

``` bash
pip install numpy
```

## Usage

To use the methods for solving SLEs provided by this project, you need to import the corresponding module and call the relevant function. Each method is implemented in a SLEsolution.py module:

    Gaussian elimination: gauss
    Square root method: squareRootMethod
    Simple iteration method: simpleIterationMethod
    Gauss-Seidel iteration: gaussSeidelMethod

For example, to solve an SLE using Gaussian elimination, you can use the following code:

```python

import numpy as np
from SLEsolution import gauss

A = np.array([[2, 3, -1], [4, 4, -3], [2, -3, 1]])
b = np.array([5, 3, 8])

x = gauss(A, b)

print(x)
```

This code creates a 3x3 matrix A and a 3x1 vector b, representing the coefficients and constant terms of the SLE, respectively. It then calls the gauss function, passing in A and b as arguments. The function returns a 3x1 vector x, representing the values of the unknown variables in the equation. Finally, the code prints the value of x to the console.
## Conclusion

This project provides a convenient way to solve systems of linear equations using several different methods. By importing the relevant module and calling the appropriate function, you can quickly and easily solve SLEs in your Python code.
