import numpy as np
import matplotlib.pyplot as plt

# -- 1D conv
n = 1000
t = np.linspace(0, 1, n)
y = np.sin(10 * t)

plt.plot(t, y)
plt.show()

# -- 2D conv
