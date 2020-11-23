# This script will implement a linear VAE for the femur dataset
# 18 November 2020
# Fabio Fehr

import os
from src.main.python.meshManipulation import loadMeshes, meshToData
from src.main.python.vae_training import trainingFunction
from sklearn.model_selection import ParameterGrid

# TODO: Make visualisations for the model (visually see how well we doing)
# TODO: Leaky relu ? Follow tan

# Experiment 1 exactly like our linear AE but with VAE
# Experiment 1 exactly like our non-linear AE but with VAE
# Experiment 2 deep use another layer
# Experiment 3  use tan's stuff
#experiment 4 train and find my own optimums?

if __name__ == '__main__':

    #CONFIG Femur
    # config = {"modelName": ["VAE", "DeepVAE"],
    #           "dataName": ["Femur"],
    #           "trainingScheme": ["LOOCV"],
    #           "activations": ["linear","tanh"],  #relu LeakyRelu
    #           "learning_rate" :[1e-4], # try a changing one
    #           "intermediate_dim" : [300], # try more
    #           "batch_size" : [10],
    #           "latent_dim" : [50],  # try 16 128 256
    #           "epochs" : [1000]  # 1000 is what i did last time
    #           }

    #CONFIG FAUST
    config = {"modelName": ["VAE", "DeepVAE"],
              "dataName": ["FAUST"],
              "trainingScheme": ["LOOCV", "People", "Poses"], # Only for FAUST
              "activations": ["linear","tanh"],  #relu LeakyRelu
              "learning_rate" :[1e-4],
              "intermediate_dim" : [300],
              "batch_size" : [25],
              "latent_dim" : [100],  # try 16 128 256
              "epochs" : [1000]  # 1000 is what i did last time
              }

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

        trainingFunction(data,
                         **configGrid[i])

    print("fin")