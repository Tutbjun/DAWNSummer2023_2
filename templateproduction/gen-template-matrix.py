#load spectres of templates at last point
import os
from astropy.table import Table
import numpy as np
import tensorflow as tf
from datetime import datetime
path = "templates-custom/45k/raw"
templates_in = [f for f in os.listdir(path) if "bin1.fits" in f or "bin0.fits" in f]
spectras = []
for temp in templates_in:
    tab = Table.read(path+"/"+temp)
    flux = tab["flux"].T[-1]
    spectras.append(flux)
spectras = np.array(spectras)

#set up matrix to opperate them together
@tf.function
def performMatrixOperation(matrix, spectra):
    return tf.matmul(matrix, spectra)

#set up orthogonalitymeassure
@tf.function
def innerProduct(vector1, vector2):
    return tf.reduce_sum(tf.multiply(vector1, vector2))

@tf.function
def orthogonalityMeassure(vector):
    arr = tf.zeros((vector.shape[0], vector.shape[0]))
    """for i in range(vector.shape[0]):
        for j in range(vector.shape[0]):
            arr[i,j] = innerProduct(vector[i], vector[j])**2 / (innerProduct(vector[i], vector[i]) * innerProduct(vector[j], vector[j]))"""
    #implement as tf function
    for i in tf.range(vector.shape[0]):
        for j in tf.range(vector.shape[0]):
            arr = tf.tensor_scatter_nd_add(arr, [[i,j]], [innerProduct(vector[i], vector[j])**2 / (innerProduct(vector[i], vector[i]) * innerProduct(vector[j], vector[j]))])
    return tf.reduce_sum(arr)


start = tf.constant(np.random.uniform(0,1, size=(13, spectras.shape[0])), dtype=tf.float32)
dummy = performMatrixOperation(start, spectras)

#set up tensorflow to optimize matrix for othogonailitymeassure
tf.debugging.set_log_device_placement(True)
tfk = tf.keras
tfkl = tf.keras.layers
tf.config.run_functions_eagerly(True)

#set up model
model = tf.Variable(start, dtype=tf.float32)

#set up loss function
@tf.function
def loss_function(model, spectra):
    dummy = performMatrixOperation(model, spectra)
    return orthogonalityMeassure(dummy)

#set up optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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

#set up training
losses = []
model = fixModel(model)
for i in range(1000):
    loss = train_step(model, spectras)
    model = fixModel(model)
    if i % 10 == 0:
        print(loss.numpy())
    if i % 30 == 0:
        np.save("templates-custom/45k/optimized_matrix.npy", model.numpy())
    print(".", end="")
    losses.append(loss.numpy())

#save model
np.save(f"templates-custom/45k/optimized_matrix_score:{loss:.3f}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npy", model.numpy())

#show losses
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()