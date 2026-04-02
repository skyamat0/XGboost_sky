import numpy as np
a = np.array([1, 2, 3, 3, 5, 5])
b = np.array([[1]])
print(a[a==3])
print(np.arange(10))
print(b.ndim)

class A:
    pass
x = [A() for i in range(5)]
print(x)