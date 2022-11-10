import numpy as np
import matplotlib.pyplot as plt

"""
    1D conv
"""
if False:
    n = 1000
    t = np.linspace(0, 1, n)
    y = np.sin(10 * t) + 0.3 * np.random.randn(n)

    plt.plot(t, y)
    # plt.show()
    # plt.close()

    # -- define filter
    t_ = np.arange(-400, 400)
    sigma = 60
    k = np.exp(-t_**2/(2*sigma**2))
    k = k/np.sum(k)



    y_conv = np.convolve(y, k, mode='same')

    plt.plot(t, y_conv)
    plt.xlim(0, 1.)
    plt.show()
    plt.close()

    plt.plot(t_, k)
    plt.show()
    plt.close()

"""
    2D conv
"""

from PIL import Image


x = np.asarray(Image.open('dome_rgb.jpg'))
x = 0.2 * x[:, :, 0] + 0.5 * x[:, :, 1] + 0.1 * x[:, :, 2]

plt.imshow(x, cmap='gray')
plt.show()
