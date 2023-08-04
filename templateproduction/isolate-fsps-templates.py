import numpy as np
from astropy.table import Table
from astropy.io import fits
import os
import shutil

dir = [f for f in os.listdir("templates-custom/45k/") if ".npy" in f]
#fileScores = [float(f.split("score:")[1].split("_")[0]) for f in dir]
#sort = np.argsort(fileScores)
#matrix = np.load("templates-custom/45k/" + dir[sort[0]])
matrix = np.load("templates-custom/45k/optimized_matrix.npy")
print(matrix.shape)

pathin = "templates-custom/45k/raw"
pathout = "templates-custom/45k/"
templates_in = [f for f in os.listdir(pathin) if "bin1.fits" in f or "bin0.fits" in f]
spectras = []
for temp in templates_in:
    tab = Table.read(pathin+"/"+temp)
    flux = tab["flux"].T[-1]
    spectras.append(flux)
spectras = np.array(spectras)

for i in range(matrix.shape[0]):
    shutil.copy(os.path.join(pathin, templates_in[0]), os.path.join(pathout, f"optimizedTemplate_{i}.fits"))
    tab = Table.read(os.path.join(pathout, f"optimizedTemplate_{i}.fits"))
    fluxes = tab["flux"].T
    continuums = tab["continuum"].T
    dereds = tab["dered"].T
    for j in range(len(fluxes)):
        fluxes[j] = np.dot(matrix[i], spectras)
        continuums[j] = np.dot(matrix[i], spectras)
        dereds[j] = np.dot(matrix[i], spectras)
    fitFile = fits.open(os.path.join(pathout, f"optimizedTemplate_{i}.fits"))
    fitFile[1].data["flux"] = fluxes.T
    fitFile[1].data["continuum"] = continuums.T#!noo clue if this is the corect way to do it
    fitFile[1].data["dered"] = dereds.T
    fitFile.writeto(os.path.join(pathout, f"optimizedTemplate_{i}.fits"), overwrite=True)
    print(f"Template {i} done")