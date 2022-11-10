import numpy as np
import matplotlib.pyplot as plt

"""
    1D conv
"""
n = 1000
t = np.linspace(0, 1, n)
y = np.sin(10 * t) + 0.3 * np.random.randn(n)

plt.plot(t, y)
plt.show()

# -- define filter
t_ = np.arange(-400, 400)
sigma = 60
k = np.exp(-t_**2/(2*sigma**2))
k = k/np.sum(k)

plt.plot(t_, k)
plt.show()


"""
    2D conv
"""
