# This script will implement a linear VAE for the femur dataset
# 18 November 2020
# Fabio Fehr

import os
from src.main.python.meshManipulation import loadMeshes, meshToData
from src.main.python.vae_training import trainingFunction
from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':

    # Experiment 1 vanilla VAE
    # Simple experiment of VAE no frills plain vanilla

    # CONFIG Femur
    # config = {"modelName": ["VAE"],
    #           "dataName": ["Femur"],
    #           "trainingScheme": ["LOOCV"],
    #           "activations": ["tanh"],  # wont be used in this experiment
    #           "learning_rate" :[1e-4],
    #           "intermediate_dim" : [300], # Wont be used in this experiment
    #           "batch_size" : [10],
    #           "latent_dim" : [50],
    #           "epochs" : [1000],
    #           "initialiser": ["glorot_uniform"]
    #           }

    # #CONFIG FAUST
    # config = {"modelName": ["VAE"],
    #           "dataName": ["FAUST"],
    #           "trainingScheme": ["LOOCV", "People", "Poses"], # Only for FAUST
    #           "activations": ["tanh"], # wont be used in this experiment
    #           "learning_rate" :[1e-4],
    #           "intermediate_dim" : [300], # Wont be used in this experiment
    #           "batch_size" : [25],
    #           "latent_dim" : [100],
    #           "epochs" : [1000],
    #           "initialiser": ["glorot_uniform"]
    #           }

    # Experiment 2 VAE deep.
    # Simple experiment of adding a hidden layers. Changing the node amounts
    #
    # CONFIG Femur
    # config = {"modelName": ["DeepVAE"],
    #           "dataName": ["Femur"],
    #           "trainingScheme": ["LOOCV"],
    #           "activations": ["leakyRelu"],
    #           "learning_rate" :[1e-4],
    #           "intermediate_dim" : [256, 512], # 256, 512
    #           "batch_size" : [10],
    #           "latent_dim" : [50,128], # 50, 128
    #           "epochs" : [1000],
    #           "initialiser": ["glorot_uniform"]
    #           }
    #CONFIG FAUST
    # config = {"modelName": ["DeepVAE"],
    #           "dataName": ["FAUST"],
    #           "trainingScheme": ["LOOCV", "People", "Poses"], # Only for FAUST
    #           "activations": ["leakyRelu"],
    #           "learning_rate" :[1e-4],
    #           "intermediate_dim" : [256, 512],
    #           "batch_size" : [25],
    #           "latent_dim" : [100,128], # perhaps 256 and 256?
    #           "epochs" : [1000],
    #           "initialiser": ["glorot_uniform"]
    #           }

    # Experiment 3 initialisations + epochs
    # Finding the better parameters
    # #CONFIG Femur
    # config = {"modelName": ["VAE"], # vanilla
    #           "dataName": ["Femur"],
    #           "trainingScheme": ["LOOCV"],
    #           "activations": ["leakyRelu"],  # doesnt matter
    #           "learning_rate" :[1e-4],
    #           "intermediate_dim" : [300], # doesnt matter
    #           "batch_size" : [10],
    #           "latent_dim" : [50],
    #           "epochs" : [250],
    #           "initialiser": ["glorot_uniform","he_normal"]
    #           }
    # Doesn't make sense to do  more epochs as is stabalised EVEN AT 100

    # #CONFIG FAUST
    # config = {"modelName": ["DeepVAE"],
    #           "dataName": ["FAUST"],
    #           "trainingScheme": ["LOOCV"],
    #           "activations": ["leakyRelu", "elu"],
    #           "learning_rate" :[1e-4],
    #           "intermediate_dim" : [256],
    #           "batch_size" : [10],
    #           "latent_dim" : [100],
    #           "epochs" : [2000],
    #           "initialiser": ["glorot_uniform","he_normal"]
    #           }

    # He initialisation, epochs?


    configGrid = list(ParameterGrid(config))

    for i in range(0, len(configGrid)):
        print(configGrid[i])
        dataName = configGrid[i]["dataName"]
        os.chdir("/media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj2_python/")

        # fetch data
        if(dataName == "Femur"):
            meshes = loadMeshes("meshes/femurs/processedFemurs/", ply_Bool=False)  # dims 50 36390
        else: # dataName == "FAUST"
            meshes = loadMeshes("meshes/faust/aligned/", ply_Bool=True)  # dims 100 36390


        # create vertices dataset
        rawData = meshToData(meshes)
        mean = rawData.mean(axis=0)

        # center the data
        data = (rawData - mean)

        # Get triangles
        triangles = meshes[0].triangles

        trainingFunction(data,triangles, mean,
                         **configGrid[i])

    print("fin")