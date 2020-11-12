# Running autoencoder pipeline for 3D meshes with non-linear activations
# Fabio Fehr
# 1 July 2020

from src.meshManipulation import loadMeshes, meshToData
from src.autoencoder_training import trainingAEViz, training_function
import glob2

def main():

    # fetch data
    meshes = loadMeshes("../meshes/faust/")

    # create vertices dataset
    rawData = meshToData(meshes)
    mean = rawData.mean(axis=0)

    # center the data
    data = (rawData - mean)

    # Get triangles
    triangles = meshes[0].triangles

    # Set colour
    colour = [205, 155, 29]  # goldenrod3

    #### TRAINING ####

    # Full training scheme
    # param_grid = {'dimension': [100],
    #               'epochs': [20000],
    #               'learning_rate': [1e-4],
    #               'batch_size': [25],
    #               'regularization': [1e-4],
    #               'activation': ["sigmoid","relu","tanh"]}

    # Best result
    param_grid = {'dimension': [100],
                  'epochs': [1000],
                  'learning_rate': [1e-4],
                  'batch_size': [25],
                  'regularization': [1e-5],
                  'activation': ["tanh"]}

    # training_function(data, param_grid, name='faust_nonlinear_')

    #### VISUALISING ####

    # Set the directory and the wild cards to select all runs of choice

    direc = '../results/'

    paths = glob2.glob(direc + "faust_nonlinear_tanh_AE_w2_*")

    trainingAEViz(rawData, paths, triangles, "faust_nonlinear_AE_", colour, cameraName="faust",eigen_faust_Bool= True,trim_type="faust")


if __name__ == '__main__':
    main()