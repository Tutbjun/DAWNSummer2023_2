import numpy as np
from astropy.table import Table
from astropy.io import fits
import os
import shutil

dir = [f for f in os.listdir("templates-custom/45k/") if ".npy" in f and "score:" in f]
fileScores = [float(f.split("score:")[1].split("_")[0]) for f in dir]
sort = np.argsort(fileScores)
matrix = np.load("templates-custom/45k/" + dir[sort[0]])
matrix = np.load("templates-custom/45k/optimized_matrix.npy")
print(matrix.shape)

#pathin = "templates-custom/45k/raw"
pathin = "templates-custom/EMlines"#!temp
#pathout = "templates-custom/45k/"
pathout = "templates-custom"#!temp
pathin = os.path.dirname(os.path.realpath(__file__)) + "/" + pathin
pathout = os.path.dirname(os.path.realpath(__file__)) + "/" + pathout
#name = "45k"
name = "EMlines"#!temp
templates_in = [f for f in os.listdir(pathin) if "bin1.fits" in f or "bin0.fits" in f or ".spec" in f]
spectras = []
for temp in templates_in:
    if ".fits" in temp:
        tab = Table.read(pathin+"/"+temp)
        flux = tab["flux"].T[-1]
    elif ".spec" in temp:
        tab = np.loadtxt(pathin+"/"+temp)
        flux = tab.T[-1]
    spectras.append(flux)
spectras = np.array(spectras)

if 'fsps_45k_alpha.param.fits' in os.listdir(pathin):
    shutil.copy(os.path.join(pathin, 'fsps_45k_alpha.param.fits'), os.path.join(pathout, 'optimized_45k.param.fits'))
    metatab_in = Table.read(os.path.join(pathin, 'fsps_45k_alpha.param.fits'))
    #remove 'file' column
    metatab = {}
    for key in metatab_in.keys():
        if key != 'file':
            metatab[key] = metatab_in[key]
    metatabNew = {}
    for key in metatab.keys():
        metatabNew[key] = []
        for i in range(matrix.shape[0]):
            matRow = matrix[i]
            metatabNew[key].append(np.dot(matRow, metatab[key]))
    metatab_out = fits.open(os.path.join(pathout, 'optimized_45k.param.fits'))
    for key in metatab_in.keys():
        for i in range(matrix.shape[0]):
            if key == "file":
                metatab_out[1].data[key][i] = f"optimizedTemplate_{i}.fits"
            else:
                metatab_out[1].data[key][i] = metatabNew[key][i]
    metatab_out.writeto(os.path.join(pathout, 'optimized_45k.param.fits'), overwrite=True)


for i in range(matrix.shape[0]):
    if '.fits' in templates_in[0]:
        shutil.copy(os.path.join(pathin, templates_in[0]), os.path.join(pathout, f"optimizedTemplate_{i}.fits"))
    else:
        shutil.copy(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates-custom', 'blankTempl.fits'), os.path.join(pathout, f"optimizedTemplate_{i}.fits"))
    tab = Table.read(os.path.join(pathout, f"optimizedTemplate_{i}.fits"))
    
    fluxes = list(tab["flux"].T)
    continuums = list(tab["continuum"].T)
    dereds = list(tab["dered"].T)
    for j in range(len(fluxes)):
        fluxes[j] = np.dot(matrix[i], spectras)
        continuums[j] = np.dot(matrix[i], spectras)
        dereds[j] = np.dot(matrix[i], spectras)
    fluxes = np.array(fluxes)
    continuums = np.array(continuums)
    dereds = np.array(dereds)
    fitFile = fits.open(os.path.join(pathout, f"optimizedTemplate_{i}.fits"))
    if len(fluxes.T) != len(fitFile[1].data["flux"]):
        fitFile[1].data = fitFile[1].data[:len(fluxes.T)]
    fitFile[1].data["flux"] = fluxes.T
    fitFile[1].data["continuum"] = continuums.T#!noo clue if this is the corect way to do it
    fitFile[1].data["dered"] = dereds.T
    fitFile.writeto(os.path.join(pathout, f"optimizedTemplate_{i}.fits"), overwrite=True)
    print(f"Template {i} done")

#gen param file
with open(f"{pathout}/optimized_{name}.param", "w") as f:
    for i in range(matrix.shape[0]):
        f.write(f"{i+1}  ./templates/{name}_linearComb/optimizedTemplate_{i}.fits\n")
f.close()