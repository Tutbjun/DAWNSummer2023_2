##### for now a simple sfh generator that uses figure 2 of the balmer break paper as inspiration
##### TODO: actually use the SPHINX results to make a representative SFH set
import numpy as np
import matplotlib.pyplot as plt
import os

step = 0.001
X = (10**np.arange(np.log10(3.e-4), np.log10(14.127) + step, step))
Ys = []

# make a constant SFH at 2.45*10⁻⁸ Msun/yr
Ys.append(np.full(len(X), 2.45e-8))

# make a constant SFH at 2.375*10⁻⁸ Msun/yr that then drops off to 0 at 0.05 Gyr
Ys.append(np.full(len(X), 2.375e-8))
Ys[-1][np.where(X > 0.05)] = 0

# make a constant SFH at 1.44*10⁻⁸ Msun/yr that then drops off to 0 at 0.07 Gyr
Ys.append(np.full(len(X), 1.44e-8))
Ys[-1][np.where(X > 0.07)] = 0

# make a gappy SFH with gaussians at 100 Myr, 200 Myr, 320 Myr, 480 Myr, 620 Myr, and then for every 500Myr with amplitude of 2*10⁻⁸ Msun/yr and sigma of 20 Myr
Ys.append(np.zeros(len(X), dtype=float))
gauss = lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
Ys[-1] += 2e-8 * gauss(X, 0.03, 0.005)
Ys[-1] += 2e-8 * gauss(X, 0.1, 0.01)
Ys[-1] += 2e-8 * gauss(X, 0.2, 0.01)
Ys[-1] += 2e-8 * gauss(X, 0.32, 0.01)
Ys[-1] += 2e-8 * gauss(X, 0.48, 0.01)
Ys[-1] += 2e-8 * gauss(X, 0.62, 0.01)
for i in range(1, 5):
    Ys[-1] += 2e-8 * gauss(X, 1*i+0.4, 0.01)


# make every SFH be 0 before 130 Myr
for i,Y in enumerate(Ys):
    Ys[i][np.where(X < 0.005)] = 0

# save them
np.savetxt(os.path.join(os.path.dirname(__file__), "sfh_custom.txt"), Ys)

# plot them
for Y in Ys:
    plt.plot(X, Y)

plt.xlim(0.01, 3)
plt.ylim(1e-9, 0.5e-7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Age (Gyr)')
plt.ylabel('SFR (Msun/yr)')
plt.show()