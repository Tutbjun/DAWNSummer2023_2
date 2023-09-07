#load spectres of templates at last point
import os
from astropy.table import Table
import numpy as np

#export os variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
num_threads = 5
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

import tensorflow as tf
from datetime import datetime

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

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
    elif ".spec" in temp:
        tab = np.loadtxt(path+"/"+temp)
        flux = tab.T[-1]
    spectras.append(flux)
spectras = np.array(spectras)
for i in range(len(spectras)):
    #normalize
    spectras[i] = spectras[i] / np.sum(spectras[i])


#set up matrix to opperate them together
@tf.function
def performMatrixOperation(matrix, spectra):
    return tf.matmul(matrix, spectra)

#set up orthogonalitymeassure
@tf.function
def innerProduct(vector1, vector2):
    return tf.reduce_sum(tf.multiply(vector1, vector2))

"""@tf.function
def orthogonalityMeassure(vector):
    arr = tf.zeros((vector.shape[0], vector.shape[0]))
    #implement as tf function
    for i in tf.range(vector.shape[0]):
        for j in tf.range(vector.shape[0]):
            arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [innerProduct(vector[i], vector[j])**2 / (innerProduct(vector[i], vector[i]) * innerProduct(vector[j], vector[j]))])
    return tf.reduce_sum(arr)"""

@tf.function
def gridOut(vector):
    #make an othogonality meassure by the sum of the deltas squared
    arr = tf.zeros((vector.shape[0], vector.shape[0]))
    for i in tf.range(vector.shape[0]):
        for j in tf.range(vector.shape[0]):
            #arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [tf.reduce_sum(tf.math.abs(vector[i] - vector[j])**2)/(innerProduct(vector[i], vector[i]) * innerProduct(vector[j], vector[j]))])
            """arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [
                
                tf.math.abs(innerProduct(vector[i], vector[j]))
                 / tf.math.sqrt(innerProduct(vector[i], vector[i]) * innerProduct(vector[j], vector[j]))
                
                ])"""
            #arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [tf.reduce_sum(tf.math.abs(vector[i] - vector[j])/((vector[i] + vector[j])/2))])
            #arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [tf.reduce_sum(tf.math.abs(vector[i] - vector[j]))])
            #arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [tf.reduce_sum(tf.math.log(tf.math.abs(vector[i] - vector[j])+1))])
            #arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [tf.reduce_sum((tf.math.abs(vector[i] - vector[j])+0.00000001)**2)])
            #arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [tf.reduce_sum(tf.math.sqrt(tf.math.abs(vector[i] - vector[j])+0.00000001))])
            arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [tf.reduce_sum(tf.math.pow(tf.math.abs(vector[i] - vector[j])+0.00000001,1.0/4.0))])
    return tf.math.sqrt(arr)

@tf.function
def vecOut(vector):
    grid = gridOut(vector)#get grid of the differences between the vectors
    #row = tf.reduce_sum(grid**2, axis=0)# square it and sum over the rows to gain the uniqueness of each vector
    #return tf.math.sqrt(row)#take the square root to get the uniqueness of each vector
    #return tf.math.sqrt(tf.reduce_sum(grid, axis=0))
    return tf.math.sqrt(tf.reduce_prod(grid, axis=0))


@tf.function
def orthogonalityMeassure(vector):
    #make an othogonality meassure by the sum of the deltas squared
    """arr = tf.zeros((vector.shape[0], vector.shape[0]))
    for i in tf.range(vector.shape[0]):
        for j in tf.range(vector.shape[0]):
            arr = -tf.tensor_scatter_nd_add(arr, [[i,j]], [tf.reduce_sum(tf.math.abs(vector[i] - vector[j]))/innerProduct(vector[i], vector[j])])"""
    #arr = gridOut(vector)
    #return tf.reduce_sum(arr)
    #return tf.reduce_sum(vecOut(vector))
    #return tf.math.sqrt(tf.reduce_sum(vecOut(vector)**2))
    return tf.reduce_sum(vecOut(vector))

@tf.function
def rowwiseStandardDeviationMeassure(model):
    var = 0
    for i in tf.range(model.shape[0]):
        var += tf.math.reduce_std(model[i])**2
    return tf.math.sqrt(var)

@tf.function
def columnwiseStandardDeviationMeassure(model):
    var = 0
    for i in tf.range(model.shape[1]):
        var += tf.math.reduce_std(model[:,i])**2
    return tf.math.sqrt(var)

#set up tensorflow to optimize matrix for othogonailitymeassure
tfk = tf.keras
tfkl = tf.keras.layers
tf.config.run_functions_eagerly(True)

#set up loss function
@tf.function
def loss_function(model, spectra):
    model = fixModel(model)
    dummy = performMatrixOperation(model, spectra)
    loss = -orthogonalityMeassure(dummy) #- rowwiseStandardDeviationMeassure(model) - columnwiseStandardDeviationMeassure(model)
    return loss

#set up optimizer
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.02)

#set up training step
@tf.function
def train_step(model, spectra):
    with tf.GradientTape() as tape:
        tape.watch(model)
        loss = loss_function(model, spectra)
    gradients = tape.gradient(loss, model)
    optimizer.apply_gradients(zip([gradients], [model]))
    return loss

#set up fix function
@tf.function
def fixModel(model):
    #if any value is negative, set it to 0
    #normalize outputs of each row
    for i in tf.range(model.shape[0]):
        for j in tf.range(model.shape[1]):
            if model[i,j] < 0:
                model[i,j].assign(0)
    for i in tf.range(model.shape[0]):
        model[i].assign(model[i] / tf.reduce_sum(model[i]))
    return model

import matplotlib.pyplot as plt
plt.ion()
figure, axs = plt.subplots(6, 1, figsize=(10,10))
axs[3].set_yscale("log")
axs[3].set_xscale("log")
#line1, = ax.imshow(np.zeros((13, spectras.shape[0])))
lSque = 0.1
dummy1 = np.zeros((13, spectras.shape[0]))-3
dummy2 = np.zeros((13,13))+0
dummy3 = np.zeros((1,13))+4
dummy1[0][0] = 0+lSque*2
dummy2[0][0] = 3
dummy3[0][0] = 6
axisImg1 = axs[0].imshow(dummy1, cmap="hot")
axisImg2 = axs[1].imshow(dummy2, cmap="hot")
axisImg3 = axs[2].imshow(dummy3, cmap="hot")
lines = [axs[i].plot([], [])[0] for i in range(3,6)]
cbar1 = figure.colorbar(axisImg1, ax=axs[0])#, values=np.linspace(0,1,100))
cbar2 = figure.colorbar(axisImg2, ax=axs[1])#, values=np.linspace(0,1,100))
cbar3 = figure.colorbar(axisImg3, ax=axs[2])#, values=np.linspace(0,1,100))
#cbar.ax.set_ylabel("Weight")
#cbar.vmax = 1
#cbar.vmin = 0
figure.canvas.draw()
#make colorbar

matrixName = input("Optimized matrix name: ")

losseses = []
orthogonalities = []
rowStds = []
colStds = []
with tf.device("cpu"):
    #TODO: setup multithreading
    try:
        #initialize model
        #model = tf.Variable(np.random.uniform(0,1, size=(13, spectras.shape[0])), dtype=tf.float32)
        model = tf.Variable(np.zeros((13, spectras.shape[0]),dtype=float), dtype=tf.float32)
        for i in tf.range(model.shape[1]):
            model[i%13,i].assign(1)
        losses = []
        model = fixModel(model)
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
plt.show()