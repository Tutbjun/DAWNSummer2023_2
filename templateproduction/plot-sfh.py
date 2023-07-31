import numpy as np
import matplotlib.pyplot as plt
import os

# Read in the data
file = "sfh_alpha_c2020.txt"
step = 0.001
X = (10**np.arange(np.log10(3.e-4), np.log10(14.127) + step, step))
Ys = np.loadtxt(os.path.join(os.path.dirname(__file__), file))

for Y in Ys:
    plt.plot(X, Y)

plt.xscale('log')
plt.xlabel('Age (Gyr)')
plt.ylabel('SFR (Msun/yr)')
plt.show()