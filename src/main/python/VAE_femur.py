from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.callbacks import TensorBoard
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
from src.main.python.meshManipulation import loadMeshes, meshToData
from src.main.python.vae_training import trainingFunction
import csv

# from autoencoder_training import trainingAEViz, training_function
import glob2

# This script will implement a VAE for the femur dataset
# 5 November 2020
# Fabio Fehr
# The template for this model is found at
# https://github.com/piyush-kgp/VAE-MNIST-Keras

# TODO: download old git repo and see if it works! if so then we can make edits from there!
# TODO: Set up single and many hidden layer options
# TODO: Make visualisations for the model (visually see how well we doing)
# TODO: Leaky relu ? Follow tan

  # light blue colour

    # #### TRAINING ####
    #
    # # Full training scheme
    # # param_grid = {'dimension': [50],
    # #               'epochs': [10000],
    # #               'learning_rate': [1e-4],
    # #               'batch_size': [5,10],
    # #               'regularization': [1e-1,1e-2, 1e-3],
    # #               'activation': ['linear']}
    #
    # # Best result
    # param_grid = {'dimension': [50],
    #               'epochs': [1000],
    #               'learning_rate': [1e-4],
    #               'batch_size': [10],
    #               'regularization': [1e-5],
    #               'activation': ['linear']}
    #
    # # training_function(data, param_grid,name='femur_linear_')
    #
    # #### VISUALISING ####
    #
    # # Set the directory and the wild cards to select all runs of choice
    #
    # direc = '../results/'
    # paths = glob2.glob(direc + "*femur_linear_linear_AE_w2*")
    # trainingAEViz(rawData, paths, triangles, "femur_linear_AE_", colour,
    #               cameraName="femur",eigen_faust_Bool=False,trim_type="femur")





# def plot_results(*args,
#                  batch_size=128,
#                  model_name="vae_mnist"):
#     """Plots labels and MNIST digits as function of 2-dim latent vector
#     # Arguments:
#         models (tuple): encoder and decoder models
#         data (tuple): test data and label
#         batch_size (int): prediction batch size
#         model_name (string): which model is using this function
#     """
#
#     encoder, decoder, x_test, y_test = args
#     os.makedirs(model_name, exist_ok=True)
#
#     filename = os.path.join(model_name, "vae_mean.png")
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = encoder.predict(x_test,
#                                    batch_size=batch_size)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
#     plt.colorbar()
#     plt.xlabel("Dimension 1")
#     plt.ylabel("Dimension 2")
#     plt.savefig(filename)
#
#     filename = os.path.join(model_name, "digits_over_latent.png")
#     # display a 30x30 2D manifold of digits
#     n = 30
#     digit_size = 28
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-4, 4, n)
#     grid_y = np.linspace(-4, 4, n)[::-1]
#
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[i * digit_size: (i + 1) * digit_size,
#                    j * digit_size: (j + 1) * digit_size] = digit
#
#     plt.figure(figsize=(10, 10))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap='Greys_r')
#     plt.savefig(filename)



if __name__ == '__main__':


    #CONFIG
    intermediate_dim = 512 # 1024 # first layer
    batch_size = 10
    latent_dim = 128 # Usually 2?
    epochs = 1 # 500 roughly 15 mins (need more)
    modelName = "VAE"
    dataName = "Femur"
    trainingScheme = "LOOCV"

    os.chdir("/media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj2_python/")
    # fetch data
    meshes = loadMeshes("meshes/femurs/processedFemurs/", ply_Bool=False)  # dims 50 36390

    # create vertices dataset
    rawData = meshToData(meshes)
    mean = rawData.mean(axis=0)

    # center the data
    data = (rawData - mean)

    # Get triangles
    triangles = meshes[0].triangles

    # Set colour
    colour = [141, 182, 205]
    # print(np.array(meshes[0].vertices).size) # 35982 femurs size which is 11994 (x,y,z)'s
    # print(data.shape[0],data.shape[1]) #50 35982

    # k = data.shape[0] # how many folds LOOCV
    k =10 #data.shape[0]

    trainingFunction(data,
                     k,
                     latent_dim,
                     intermediate_dim,
                     batch_size,
                     epochs,
                     modelName,
                     dataName,
                     trainingScheme)

    print("fin")