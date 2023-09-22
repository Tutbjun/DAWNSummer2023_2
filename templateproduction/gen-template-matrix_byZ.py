#!/usr/bin/env python
# coding: utf-8

# In[45]:


#load spectres of templates at last point
import os
from astropy.table import Table
import numpy as np
import eazy
import matplotlib.pyplot as plt
from numba import jit
from astropy.io import fits
import helper_module as hmod
from astropy.table import Table, join
import eazy_routines as ez
from astropy import units as u
import time
from astropy.cosmology import Planck18
import pandas as pd
import shutil
import sys, os
import warnings
from scipy.optimize import curve_fit
#warnings.filterwarnings("ignore")

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# In[46]:


runTime = int(time.time())
__file__ = os.getcwd()+'/'+"gen-template-matrix_byZ.ipynb"
cosmo = Planck18
set2Limit = "EMlines"


# In[47]:


############# SETTINGS #############
templCnt = 13
cCap = 0.5
doMultithread = False
threadCnt = 4
trainsetCap = 150
####################################
if not doMultithread: threadCnt = 1


# In[48]:


#path = "templates-custom/45k/raw"
path = "templates-custom/EMlines"#!temp
#out_path = "templates-custom/45k"
out_path = "templates-custom"#!temp
path = os.path.dirname(os.path.realpath(__file__)) + "/" + path
out_path = os.path.dirname(os.path.realpath(__file__)) + "/" + out_path

templates_in = [f for f in os.listdir(path) if "bin1.fits" in f or "bin0.fits" in f or ".spec" in f]
spectras = []
for temp in templates_in:
    if ".fits" in temp:
        tab = Table.read(path+"/"+temp)
        flux = tab["flux"].T[-1]
        wave = tab["wave"].T[0]
    elif ".spec" in temp:
        tab = np.loadtxt(path+"/"+temp)
        flux = tab.T[-1]
        wave = tab.T[0]
    spectras.append(flux)
spectras = np.array(spectras)
for i in range(len(spectras)):
    #normalize
    spectras[i] = spectras[i] / np.sum(spectras[i])


# In[49]:


############# FILTERS #############
flt = eazy.filters.FilterFile()

filts_nircam = {
        'F090W': 363,
        'F115W': 364,
        'F150W': 365,
        'F182M': 370,
        'F200W': 366,
        'F210M': 371,
        'F277W': 375,
        'F335M': 381,
        'F356W': 376,
        'F410M': 383,
        'F430M': 384,
        'F444W': 358,
        'F460M': 385,
        'F480M': 386
}

filts_HST = {
        'F105W': 202,
        'F125W': 203,
        'F140W': 204,
        'F160W': 205,
        'F435W': 233,
        'F606W': 214,
        'F775W': 216,
        'F814W': 239,
        'F850LP': 240
}

filts = {**filts_nircam, **filts_HST}

mw_reddening = ez.get_atten_dict(filts)

# get zeropoints
zps = [1.0]*len(filts)


# In[50]:


############# data loading #############
#load data
inname = "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v1.0_catalog_large_withSpec.fits"
inpath = os.path.join(os.getenv('astrodata'), 'gds/jades/phot', inname)

# print the meta info
with fits.open(inpath) as hdul:
    print(hdul.info())

# load photometry table
tab = Table.read(inpath, hdu=6)
tab_redshifts = Table.read(inpath, hdu=9)

# load fluxes
# CIRC1: 0.10 arcsec aperture (see README)
ext = '_CIRC1'
cols_fluxes = hmod.get_matches(ext, tab.columns, exclude='_ei')
cols_f = np.sort(hmod.get_matches(ext, cols_fluxes, exclude='_e'))
cols_fe = np.sort(hmod.get_matches('_e', cols_fluxes))
cols_fluxes = list(np.vstack([cols_f, cols_fe]).T.flatten())
cols = list(np.insert(cols_fluxes, 0, ['ID', 'RA', 'DEC', 'z_spec','z_spec_quality']))

tab = join(tab, tab_redshifts['ID', 'z_spec', 'z_spec_quality'], join_type='inner', keys='ID')
tab_out = tab[cols]

# convert from nJy to uJy
# and apply MW reddening
keys = np.array(list(mw_reddening.keys()))
for c in cols_fluxes:
    tab_out[c].unit = u.nJy
    tab_out[c] = tab_out[c].to(u.uJy)
    
    # apply MW reddening
    matches = hmod.get_matches(keys, c, get_idxs=True)
    key = keys[np.int32(matches[:,0])][0]
    tab_out[c] *= mw_reddening[key]

# rename columns
for c in cols_f:
    cnew = c.replace(ext, '_flux')
    tab_out.rename_column(c, cnew)

for c in cols_fe:
    cnew = c.replace(ext+'_e', '_err')
    tab_out.rename_column(c, cnew)

# pick out training set
tab_out_Aqual = tab_out[tab_out['z_spec_quality'] == 'A']

#np.random.seed(42)#pick randomly 90% of the A-quality spectra as the training set
tab_out_Aqual_train = tab_out_Aqual[np.random.choice(range(len(tab_out_Aqual)), min(int(len(tab_out_Aqual)*0.9),200), replace=False)]
tab_out_Aqual_test = tab_out_Aqual[~np.isin(tab_out_Aqual['ID'], tab_out_Aqual_train['ID'])]

# save EAZY table
os.makedirs('data', exist_ok=True)
tab_out.write('./data/gds_jades_eazy.fits', format='fits', overwrite=True)
tab_out_Aqual.write('./data/gds_jades_eazy_Aqual.fits', format='fits', overwrite=True)
tab_out_Aqual_train.write('./data/gds_jades_eazy_Aqual_train.fits', format='fits', overwrite=True)
tab_out_Aqual_test.write('./data/gds_jades_eazy_Aqual_test.fits', format='fits', overwrite=True)


# In[51]:


#setup intermediate folderstructure and such

def saveTemplateset(templateSpectras):
    #save the new templateset
    for i in range(len(templateSpectras)):
        if f"{i}.spec" in os.listdir(intermediateFolder):
            os.remove(f'{intermediateFolder}/{i}.spec')
            #print("Removed intermediate file: "+f'{intermediateFolder}/{i}.spec')
        flux = templateSpectras[i]
        writeList = np.array([wave, flux]).T
        np.savetxt(f'{intermediateFolder}/{i}.spec', writeList)
        #print("Created template file: "+f'{intermediateFolder}/{i}.spec')
    print("updated template set")

intermediateFolder = "spectras_temp"
if not os.path.exists(intermediateFolder):
    os.makedirs(intermediateFolder)
    print("Created intermediate folder: "+intermediateFolder)
if os.path.exists(f'{intermediateFolder}/paramFile.param'):
    os.remove(f'{intermediateFolder}/paramFile.param')
    print("Removed intermediate file: "+f'{intermediateFolder}/paramFile.param')
with open(f'{intermediateFolder}/paramFile.param', 'w') as p:
    saveTemplateset(spectras[:templCnt])
    for i in range(templCnt):
        p.write(f'{i+1} {intermediateFolder}/{i}.spec\n')
p.close()


# In[52]:


#=== set up paths for eazy

# catalog paths
cat_name = 'gds_jades_eazy_Aqual_train'
cat_path = f'./data/{cat_name}.fits'
keys_id = ['ID id', 'RA ra', 'DEC dec', 'z_spec z_spec']

templ_paths = f'{intermediateFolder}/paramFile.param'


# In[53]:


@jit(nopython=True)
def performMatrixOperation(matrix, spectra):
    return np.dot(matrix,spectra)

def runEAZY():
    tpath = f'{intermediateFolder}/paramFile.param'
    with open(tpath) as f:
        lines = f.readlines()
        f.close()
    opath = f'./eazy-output/{runTime}'
    params = {"cat_path": cat_path,
        "templ_path": tpath,
        "out_path": opath,
        "FIX_ZSPEC": 'n',
        "USE_ZSPEC_FOR_REST": 'n',
        "Z_MAX": 15,
        "H0": cosmo.H0,
        "OMEGA_M": cosmo.Om0,
        "OMEGA_L": cosmo.Ode0,
        "CATALOG_FORMAT": 'fits'}
    # write eazy config files
    ___, fnames = ez.write_config(cat_name, filts, zps, keys_id,
        out_path=opath)
    # run eazy
    idx = None
    with HiddenPrints():
        #blockPrint()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, __ = ez.run_eazy(params, fnames, n_proc=-1, idx=idx)
            warnings.resetwarnings()
        #enablePrint()
    #zout, hdu = 
    #read the output
    fpath = f'eazy-output/{runTime}/gds_jades_eazy_Aqual_train.zout.fits'
    tbl = Table.read(fpath)
    names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
    df = tbl[names].to_pandas()
    #change df 'id' to 'ID'
    df.rename(columns={'id': 'ID'}, inplace=True)

    #merge with catalog zspec
    fpath = './data/gds_jades_eazy_Aqual_train.fits'
    tbl = Table.read(fpath, hdu=1)    
    names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
    names = [name for name in names if 
        'ID' in name or
        'EAZY_z_a' in name or 
        'z_spec_source' in name or 
        'z_spec_quality' in name or
        'z_spec_reference' in name
    ]
    df_spec = tbl[names].to_pandas()
    
    # add spec info to the photo table
    df = pd.merge(df, df_spec, on='ID', how='left')

    print("################ EAZY DONE ###################")
    return df

def meassureBadnessOfZ(matrix,spectras):
    #function that encapsulates the whole framework of testing EAZY on a set of spectras
    templateSpectras = performMatrixOperation(matrix,spectras)
    saveTemplateset(templateSpectras)
    Zs = runEAZY()
    z_spec = Zs['z_spec']
    z_phot = Zs['z_phot']
    zDeltas = z_spec - z_phot
    return np.sum(zDeltas**2)


# In[54]:


#matrix stuff
#matrix = np.random.uniform(0, 1, (templCnt, spectras.shape[0]))
matrix = np.zeros((templCnt, spectras.shape[0]), dtype=np.float64)
for i in range(templCnt):
    matrix[i][i] = 1
usePrevMatrix = True
if usePrevMatrix:
    matrix = np.loadtxt(f'{out_path}/smartsmartsmartbetter_147.csv', delimiter=',')
    print("Loaded previous matrix")

@jit(nopython=True)
def matrixOp(matrix, spectras):
    vec = np.zeros((templCnt, 1))
    for i in range(templCnt):
        vec[i] = np.sum(matrix[i] * spectras[i])
    return vec


# In[55]:


############# MAIN #############
#setup updating plot
"""plt.ion()
figure, axs = plt.subplots(2, 1, figsize=(10,10))
axs[1].set_yscale("log")
axs[1].set_xscale("log")
dummy1 = np.zeros((13, spectras.shape[0]))-3
dummy1[0][0] = 0.1
axisImg1 = axs[0].imshow(dummy1, cmap="hot")
lines = [axs[i].plot([], [])[0] for i in range(1,2)]
cbar1 = figure.colorbar(axisImg1, ax=axs[0])#, values=np.linspace(0,1,100))
figure.canvas.draw()"""
c = 0
polynomium = lambda x, a, b, c: a*x**2 + b*x + c
#put in name for computed matrix
matrixName = input("Optimized matrix name: ")
print("################### START #######################")
def testC(c,N,matrix,spectras):
    matrix = matrix + c*N
    return meassureBadnessOfZ(matrix,spectras)
losses_train = []
losses_test = []
print("initial loss test...")
losses_train.append(testC(0,matrix,matrix,spectras))#initial loss
print("initial loss test done")
#main loop
for i in range(100000):
    print(f"loop {i}")
    betterFound = False
    def found(loss):
        global betterFound
        global losses_train
        print("IMPROVEMENT FOUND!!!")
        print(rf"$\Delta$ loss: {loss-losses_train[-1]}")
        betterFound = True
        losses_train.append(loss)
    pickBuffer = []
    if i == 0: 
        for j in range(templCnt): pickBuffer.append([j,j])
    for nCol,NCnt in enumerate(range(1,spectras.shape[0])):#nesting of c-optimization
        #set N to a matrix with a random index =1
        N = np.zeros((templCnt,spectras.shape[0]),dtype=np.float64)
        j = 0
        p1,p2 = 0,0
        while j < 1 or [p1,p2] in pickBuffer:
            p1 = np.random.randint(0,templCnt)
            #p2 = np.random.randint(0,spectras.shape[0])
            p2 = nCol
            N[p1][p2] = 1
            j += 1
        cMin = (-matrix[p1][p2])*cCap
        cMax = (1-matrix[p1][p2])*cCap
        pickBuffer.append([p1,p2])
        plt.imshow(matrix + c*N, cmap="hot")
        plt.colorbar()
        plt.savefig(f"matrix_{i}.png")
        plt.savefig("matrix_current.png")
        plt.close()
        """for j in range(len(N)):
            for k in range(len(N[j])):
                if N[j][k] > (1-matrix[j][k])/cCap: N[j][k] = (1-matrix[j][k])/cCap
                if N[j][k] < (0-matrix[j][k])/cCap: N[j][k] = (0-matrix[j][k])/cCap"""
        lossByCFunc = [[0,losses_train[-1]]]
        ###setup c plot####
        #plt.ion()
        
        
        
        def updatePlt(lossByCFunc, poly=None):
            figure, axs = plt.subplots(1, 1, figsize=(5,5))
            lines = [
                axs.plot([], [])[0],# polynomialfit
                axs.plot([], [], marker=".", linestyle="")[0]# lossByCFunc
            ]
            axs.set_xlabel("c")
            axs.set_ylabel("loss")
            lines[1].set_data(np.array(lossByCFunc)[:,0], np.array(lossByCFunc)[:,1])
            axs.set_ylim([losses_train[-1], (max(np.array(lossByCFunc)[:,1])-min(np.array(lossByCFunc)[:,1]))*1.1+min(np.array(lossByCFunc)[:,1])])
            axs.set_xlim([min(np.array(lossByCFunc)[:,0])*1.1, max(np.array(lossByCFunc)[:,0])*1.1])
            if poly is not None:
                lines[0].set_data(np.linspace(-cCap,cCap,100), np.polyval(poly, np.linspace(-cCap,cCap,100)))
            #figure.canvas.draw()
            plt.savefig(f"lossByCFunc_{i}_{NCnt}.png")
            plt.savefig("lossByCFunc_current.png")
            plt.close()
        updatePlt(lossByCFunc)
        ####################

        #initially test at pm cCap
        
        if cMax == 0:
            c = cMin/2
            print("cMin/2")
        else:
            c = cMax
            print("cMax")
        print(f"test @ cCap: {c}...")
        loss = testC(c,N,matrix,spectras)
        print("first c test done")
        if loss < losses_train[-1]:
            found(loss)
            cCnt = 1
            #break
            continue
        lossByCFunc.append([c,loss])
        updatePlt(lossByCFunc)
        if cMin == 0:
            c = cMax/2
            print("cMax/2")
        else:
            c = cMin
            print("cMin")
        print(f"test @ -cCap: {c}...")
        loss = testC(c,N,matrix,spectras)
        if loss < losses_train[-1]: 
            cCnt = 2
            found(loss)
            #break
            continue
        lossByCFunc.append([c,loss])
        updatePlt(lossByCFunc)
        print("test at -cCap done")
        np.savetxt(f'{out_path}/{matrixName}_{i}.csv', matrix, delimiter=',')
        #find c's by solving polynomials for ))st likely bottom
        for cCnt in range(3,6):#loop of finding c's
            print(f"################### loop {cCnt-3} #######################")
            np.savetxt(f'{out_path}/{matrixName}_{i}.csv', matrix, delimiter=',')
            order = cCnt - 2
            if order < 3: order = 2
            #do polynomial fit
            #poly = curve_fit(polynomium, np.array(lossByCFunc)[:,0], np.array(lossByCFunc)[:,1], p0=[1,0,1])[0]
            poly = np.polyfit(np.array(lossByCFunc)[:,0], np.array(lossByCFunc)[:,1], deg=order)
            print(f"poly: {poly}")
            #insert dots and polyfit
            updatePlt(lossByCFunc, poly)
            #differentiate
            polyDif = np.polyder(poly)
            #find roots
            roots = np.real(np.roots(polyDif))#!exclude complex roots
            #pick a root at random but weighted by how isolated the root is and how low the loss is at that point
            propabilities = []
            for r in roots:
                closestPoints = np.array(lossByCFunc)[:,0] - r
                closestPoints = np.abs(closestPoints)
                closestPoints = np.sort(closestPoints)[:2]
                propabilities.append(np.sum(abs(closestPoints-r)))
            propabilities = np.array(propabilities)
            propabilities = propabilities / np.sum(propabilities)
            for i in range(len(propabilities)):
                propabilities[i] = propabilities[i]/(np.polyval(poly, roots[i]))
            #remove negative propabilities
            propabilities[propabilities < 0] = 0
            propabilities = propabilities**2
            propabilities = propabilities / np.sum(propabilities)
            c = np.random.choice(roots, p=propabilities)
            c = roots[np.argmin(np.polyval(poly, roots))]
            rootVal = np.polyval(poly, c)
            print(rootVal)
            """if abs(c) >= cCap:
                prob = (np.max(np.polyval(poly, np.linspace(-cCap, cCap, 100)))-np.polyval(poly, np.linspace(-cCap, cCap, 100)))
                c = np.random.choice(np.linspace(-cCap, cCap, 100), 1, p=prob/np.sum(prob))"""
            if c < 0 or c > cCap or rootVal > losses_train[-1]:#i <= 5 and c < 0: 
                prob = (np.max(np.polyval(poly, np.linspace(0, cCap, 100)))-np.polyval(poly, np.linspace(0, cCap, 100)))
                c = np.random.choice(np.linspace(0, cCap, 100), 1, p=prob/np.sum(prob))[0]
            print(f"trying c: {c}")
            loss = testC(c,N,matrix,spectras)
            lossByCFunc.append([c,loss])
            if loss < losses_train[-1]: 
                found(loss)
                break
            plt.close()
            plt.imshow(matrix + c*N, cmap="hot")
            plt.colorbar()
            plt.savefig(f"matrix_{i}.png")
            plt.savefig("matrix_current.png")
            plt.close()
            plt.imshow(N, cmap="hot")
            plt.colorbar()
            plt.savefig(f"N_{i}.png")
            print(f"loss: {loss}, c: {c}")
        updatePlt(lossByCFunc,poly)
        np.savetxt(f'{out_path}/{matrixName}_{i}.csv', matrix, delimiter=',')
        """if betterFound: 
            break"""
    if not betterFound: cCap = cCap*0.5
    if betterFound: losses_train.append(loss)
    plt.close()
    np.savetxt(f'{out_path}/{matrixName}_{i}.csv', matrix, delimiter=',')
    matrix = matrix + c*N
    for j in range(len(matrix)):
        for k in range(len(matrix[j])):
            if matrix[j][k] > 1: matrix[j][k] = 1
            if matrix[j][k] < 0: matrix[j][k] = 0
        matrix[j] = matrix[j] / np.sum(matrix[j])
    plt.imshow(matrix, cmap="hot")
    plt.colorbar()
    plt.savefig(f"matrix_{i}.png")
    plt.show()
    np.savetxt(f'{out_path}/{matrixName}_{i}.csv', matrix, delimiter=',')
    if i % 10 == 0:
        losses_test.append(meassureBadnessOfZ(matrix,spectras))
        plt.plot(losses_train, label="train")
        plt.plot(losses_test, label="test")
        plt.legend()
        plt.savefig(f"losses.png")
    print("################### loop done #######################")
    #save plot
    #plt.savefig(f"lossByCFunc_{i}.png")
    plt.close()
    if cCap == 0: break



# In[ ]:


"""
        #initialize model
        #model = tf.Variable(np.random.uniform(0,1, size=(13, spectras.shape[0])), dtype=tf.float32)
        model = tf.Variable(np.zeros((13, spectras.shape[0]),dtype=float), dtype=tf.float32)
        for i in tf.range(model.shape[1]):
            model[i%13,i].assign(1)
        losses = []
        for i in range(10000):
            loss = train_step(model, spectras)
            model = fixModel(model)
            if i % 10 == 0:
                print(loss.numpy())
            if i % 1 == 0 and i > 0:
                #do a matplotlib heatmap of matrix
                axisImg1.set_data(np.log(model.numpy()+lSque))
                dummy = performMatrixOperation(model, spectras)
                axisImg2.set_data(np.log(gridOut(dummy).numpy()+lSque))
                axisImg3.set_data(np.reshape(np.log(vecOut(dummy).numpy()), (1,13)))
                #axisImg2.set_data(gridOut(dummy).numpy()+lSque)
                #axisImg3.set_data(np.reshape(vecOut(dummy).numpy(), (1,13)))
                lines[0].set_data(np.arange(len(orthogonalities)), orthogonalities)
                lines[1].set_data(np.arange(len(rowStds)), rowStds)
                lines[2].set_data(np.arange(len(colStds)), colStds)
                axs[3].set_xlim(0, len(orthogonalities))
                axs[4].set_xlim(0, len(rowStds))
                axs[5].set_xlim(0, len(colStds))
                axs[3].set_ylim(np.min(orthogonalities), np.max(orthogonalities))
                axs[4].set_ylim(np.min(rowStds), np.max(rowStds))
                axs[5].set_ylim(np.min(colStds), np.max(colStds))
                figure.canvas.draw()
                figure.canvas.flush_events() 
            if i % 30 == 0:
                np.save(f"templates-custom/45k/optimized_matrix_{matrixName}.npy", model.numpy())
            print(".", end="")
            losses.append(loss.numpy())
            orthogonalities.append(orthogonalityMeassure(performMatrixOperation(model, spectras)))
            rowStds.append(rowwiseStandardDeviationMeassure(model))
            colStds.append(columnwiseStandardDeviationMeassure(model))
            if len(losses) > 1000:
                if abs(loss-losses[-100]) < 0.000005:
                    break
        #save model
        np.save(f"{path}/optimized_matrix_{matrixName}_score:{loss:.3f}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npy", model.numpy())
        losseses.append(losses)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        np.save(f"{path}/optimized_matrix_{matrixName}_score:{loss:.3f}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npy", model.numpy())
        losseses.append(losses)
        

#show losses
import matplotlib.pyplot as plt
[plt.plot(losses) for losses in losseses]
plt.show()"""

