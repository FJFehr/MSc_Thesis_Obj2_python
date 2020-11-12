# A collection of functions for training, visualising and saving autoencoders for 3D mesh models
# Fabio Fehr
# 1 July 2020

import numpy as np
import time
from src.meshManipulation import PlotModesVaration,\
    modesOfVariationVis, mean3DVis, \
    PlotScatterGram,shapeParameters
from keras.layers import Input, Dense
from keras import regularizers, models, optimizers
from sklearn.model_selection import ParameterGrid

# Autoencoder
def AE(y, dimension=100, learning_rate = 1e-4, regularization = 1e-4,batch_size=25, epochs=10, activation = 'linear'):

    '''
    This creates the basic frameowrk for a single layer AE to be used for
    :param y: data
    :param dimension: bottleneck hidden layer dimension
    :param learning_rate: for the adam optimiser
    :param regularization: regularisation parameter
    :param epochs: how many times the model sees the data
    :param activation: This is the decoder activation function linear or sigmoid
    :return: The weights and biases, loss per epoch, time elapsed
    '''

    start_time = time.time()

    input = Input(shape=(y.shape[1],))
    encoded = Dense(dimension, activation='linear',
                    kernel_regularizer=regularizers.l2(regularization))(input)
    decoded = Dense(y.shape[1], activation=activation,
                    kernel_regularizer=regularizers.l2(regularization))(encoded)
    autoencoder = models.Model(input, decoded)
    autoencoder.compile(optimizer=optimizers.adam(lr=learning_rate), loss='mean_squared_error')
    history = autoencoder.fit(y, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose =1)

    (w1,b1,w2,b2)=autoencoder.get_weights()

    # record time
    time_list = []
    end_time = (time.time() - start_time)
    time_list.append(end_time)

    return (w1,b1,w2,b2,history.history['loss'],time_list)


def train_AE_save(data,
                  dimension,
                  epochs,
                  learning_rate,
                  regularization,
                  batch_size,
                  activation,
                  name):
    '''
    This function takes in all the parameters for a AE and saves
     decoder weights, loss per epoch and the time taken to train in seconds

    :param data: data
    :param dimension: bottleneck hidden layer dimension
    :param learning_rate: for the adam optimiser
    :param regularization: regularisation parameter
    :param epochs: how many times the model sees the data
    :param activation: The activation type linear or sigmoid
    :param name: saving name
    :return: The weights and biases
    '''

    #Edit name for saving
    name = name + str(activation)

    # Run linear AE and return and save the weights
    (_, _, w2, _,loss,end_time) = AE(y= data,
                             dimension=dimension,
                             epochs=epochs,
                             learning_rate=learning_rate,
                             batch_size =batch_size,
                             regularization=regularization,
                             activation=activation)

    # Save loss
    np.savetxt('../results/' + name +
               '_AE_loss_dim_' + str(dimension) +
               "_reg_" + str(regularization) +
               '_epoch_' + str(epochs) +
               '_lr_' + str(learning_rate) +
               '_bs_' + str(batch_size) +
               '.csv',
               loss, delimiter=',')

    # Save time
    np.savetxt('../results/' + name +
               '_AE_time_dim_' + str(dimension) +
               "_reg_" + str(regularization) +
               '_epoch_' + str(epochs) +
               '_lr_' + str(learning_rate) +
               '_bs_' + str(batch_size) +
               '.csv',
               end_time, delimiter=',')

    # Save decider weights
    np.savetxt('../results/' + name +
               '_AE_w2_dim_'+str(dimension) +
               "_reg_" + str(regularization) +
               '_epoch_' + str(epochs) +
               '_lr_' + str(learning_rate) +
               '_bs_' + str(batch_size) +
               '.csv',
               w2, delimiter=',')


def training_function(data, param_grid, name):
    '''
    This function takes in data, a name and a parameter grid as seen below

    Here is an example parameter grid
    param_grid = {'dimension': [100],
               'epochs': [10000],
               'learning_rate': [1e-4],
                'batch_size': [5],
                'regularization': [1e-4],
                'activation': ['linear']}

    :param data: Your 3D mesh data
    :param param_grid: a dict of your parameters to run.
    :param name: The specified name
    :return: saves it
    '''

    # param_grid = {'dimension': [100],  # maybe try 20
    #               'epochs': [10000],
    #               'learning_rate': [1e-4],  # maybe 1e-6
    #               'batch_size': [5],
    #               'regularization': [1e-4]}  # maybe 1e-2

    params = list(ParameterGrid(param_grid))
    # loop through all options
    for i in range(0, len(params)):

        # Record times
        start_time = time.time()

        #show parameters
        for key, value in params[i].items():
            print(key,value)

        # train and save the AE weights
        train_AE_save(data=data,
                      name=name,
                      **params[i])

        print("completed " + str(i/len(params)))
        print("--- %s seconds ---" % +(time.time() - start_time))

def trainingAEViz(data, paths,triangles,name,col,cameraName,eigen_faust_Bool= True,trim_type=""):

    '''
    Once training has been run you will want to visualise the results

    :param data: your data
    :param paths: The paths to your training results
    :param triangles: The triangles for your mesh visualisations
    :param name: Name you will use to save.
    :param: col: [255,255,255] colour code
    :param: eigen_faust_Bool: boolean to fetch the eigen values for the faust data or the femur data
    :return: saves plots
    '''

    #Calculate the mean
    mean = data.mean(axis=0)
    mean3DVis(data, triangles, name, col,cameraName=cameraName)

    # Center data
    data = data - mean

    for path in paths:
        print(path)

        #load the path to training output
        w2 = np.loadtxt(str(path), delimiter=',')

        # Now these are equivalent to the principal components. svd on decoder weights
        (u_vectors, singular_values, _) = np.linalg.svd(w2.T, full_matrices=False)

        # Get the components
        components = u_vectors.T

        # turn to shape parameters
        b = shapeParameters(data, components)
        np.savetxt('../results/'+name+'ShapeParamaters_b.csv', b, delimiter=',')

        # Load eigenvalues from PCA for bounds
        if(eigen_faust_Bool):
            pca_eigen_values = np.loadtxt('../results/faust_PCA_Eigen.csv', delimiter=',')
        else:
            pca_eigen_values = np.loadtxt('../results/femur_PCA_Eigen.csv', delimiter=',')

        # Save the modes of variations pictures
        modesOfVariationVis(mean, components, pca_eigen_values, 3, triangles, name, col = col,cameraName=cameraName)

        # Combine them
        PlotModesVaration(3, name,trim_type=trim_type)

        # Plot basic scattergram
        PlotScatterGram(b, 3, name)