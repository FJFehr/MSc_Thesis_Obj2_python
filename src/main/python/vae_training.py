# A collection of functions for training, visualising and saving VAE for 3D mesh models
# Fabio Fehr
# 11 November 2020

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse
from keras import optimizers
from keras import backend as K
from keras.callbacks import TensorBoard
from time import time
import numpy as np
import csv


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1] # Returns the shape of tensor or variable as a tuple of int or None entries.
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon # layer object

def encoder_model(inputs,latent_dim):

    z_mean = Dense(latent_dim, name='z_mean')(inputs)
    z_log_var = Dense(latent_dim, name='z_log_var')(inputs)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder, z_mean, z_log_var


# build decoder model
def decoder_model(latent_dim, original_dim, decoder_activation_func):

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    outputs = Dense(original_dim,
                    activation=decoder_activation_func)(latent_inputs)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder

def deep_encoder_model(inputs,latent_dim, intermediate_dim,encoder_activation_func):

    x = Dense(intermediate_dim, activation=encoder_activation_func)(inputs) # layer 1
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder, z_mean, z_log_var


# build decoder model
def deep_decoder_model(latent_dim, intermediate_dim, original_dim,decoder_activation_func1,decoder_activation_func2):
    # input layer
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation=decoder_activation_func1)(latent_inputs) # layer 1
    outputs = Dense(original_dim, activation=decoder_activation_func2)(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder

def pointwiseDistance(dat1, dat2):

    # current dimension usually a long vector
    currentDim = dat1.shape[1]

    # get square difference an reshape
    sqrDiffDat = (dat1 - dat2)**2
    sqrDiffDat = np.reshape(sqrDiffDat, [int(currentDim/3), 3])

    # get point wise sums (row sum)
    pointwise_sqrDiffSum = np.sum(sqrDiffDat,axis =1)

    # Get pointwise euclidean distances
    pointwise_distances = np.sqrt(pointwise_sqrDiffSum)
    return pointwise_distances

def generalisation(model, testingData):
    # Shouldnt be smaller than 0.002 as thats a GPMM for femurs for a kak VAE

    allAverageDistances = []
    allHausdorffDistances = []

    # Loop through all items in test data (LOOCV =1 but k>1)
    for i in range(0, testingData.shape[0]):

        # item i
        current_mesh = np.reshape(testingData[i][:],[1,testingData.shape[1]])
        # models prediction of item i
        # _, _, z = encoder.predict(current_mesh)
        #
        # prediction_mesh = decoder.predict(z)

        prediction_mesh = model.predict(current_mesh)

        # get distance
        distance = pointwiseDistance(current_mesh, prediction_mesh)

        # get hausdorff
        hausdorff_distance = np.max(distance)
        allHausdorffDistances.append(hausdorff_distance)

        # get average
        average_distance = np.average(distance)
        allAverageDistances.append(average_distance)

    # for all training sets lets average over them
    totalAverageDistances = sum(allAverageDistances) / len(allAverageDistances)
    totalHausdorffDistances = sum(allHausdorffDistances) / len(allHausdorffDistances)
    return totalAverageDistances, totalHausdorffDistances


def specificity(decoder, testingData,latent_dim, numberOfSamples= 20): # Training or testing data?

    all_min_distances = []
    # Loop through number of samples
    for i in range(0, numberOfSamples):

        # Since the KL divergence ensures that the encoder maps as
        # close to a standard normal distribution as possible,
        # we can sample from a z-dimensional standard normal distribution
        # and feed it to the decoder to generate new shapes.
        z_sample = np.random.normal(0,1,latent_dim)
        z_sample = np.reshape(z_sample, [1,latent_dim])

        # get a random sample from VAE
        sample_mesh = decoder.predict(z_sample)

        all_average_distances = []
        # loop through all testing data
        for i in range(0, testingData.shape[0]):
            # compute average distance between test mesh and sample
            current_mesh = np.reshape(testingData[i][:],[1,testingData.shape[1]])

            # get distance
            distance = pointwiseDistance(current_mesh, sample_mesh)
            # get average distance
            average_distance = np.average(distance)
            all_average_distances.append(average_distance)

        min_average_distance = min(all_average_distances)
        all_min_distances.append(min_average_distance)

    return np.average(all_min_distances)

def trainingFunction(data, **config):

    # Define the parameters from the config
    latent_dim = config["latent_dim"]
    intermediate_dim = config["intermediate_dim"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    activations = config["activations"]
    modelName = config["modelName"]
    dataName = config["dataName"]
    trainingScheme = config["trainingScheme"]

    if(trainingScheme == "LOOCV"):
        k = data.shape[0]
    else:
        k = 10 # poses or people

    foldsize = int(data.shape[0] / k)  # how big the fold

    dataStore = []
    all_idx = np.arange(0, data.shape[0])
    for i in range(0, k):

        # Split the data into train and test ###########################################################################
        if(trainingScheme == "Poses"): # note! k must = 10
            # Create indecies
            lst = []
            for j in range(0, k):
                lst.append(j*k+i)

            testing_idx = np.asarray(lst)
            training_idx = np.delete(all_idx, testing_idx)

        else: # this works for LOOCV or People
            # Create indecies
            testing_idx = all_idx[i * foldsize:i * foldsize + foldsize]
            training_idx = np.delete(all_idx, testing_idx)

        # Get training and testing for CV
        testing_data = data[testing_idx][:]
        train_data = data[training_idx][:]

        # train model
        original_dim = data.shape[1]

        # Create model #################################################################################################
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=(original_dim,), name='encoder_input')

        # initiate the VAE model
        if(modelName == "DeepVAE"):
            encoder, z_mean, z_log_var = deep_encoder_model(inputs, latent_dim,intermediate_dim,"linear")
            # build decoder model
            decoder = deep_decoder_model(latent_dim, intermediate_dim, original_dim,
                                         activations,
                                         activations)

        else: # Otherwise VAE
            encoder, z_mean, z_log_var = encoder_model(inputs, latent_dim)
            # build decoder model
            decoder = decoder_model(latent_dim,  original_dim, activations)
            # instantiate VAE model

        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name=dataName+modelName)

        # print(encoder.summary())
        # print(decoder.summary())
        # print(vae.summary())

        # Creating the loss function MSE and KL divergence
        reconstruction_loss = mse(inputs, outputs)
        # reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        # Femur = 5 steps per epoch as we have a batch size of 10. meaning 1000 epochs = 5000 steps
        # 500 epochs meaning 2500 steps

        vae.compile(optimizer=optimizers.adam(lr=learning_rate))
        # default is lr =0.001 (tan), beta_1=0.9, beta_2=0.999. Could try lr = 0.0001 (Bagautdinov)

        # Train the model ##############################################################################################
        training_start_time = time()
        vae.fit(train_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(testing_data, None)
                )
        training_end_time = time()
        total_training_time = training_end_time - training_start_time

        # calculate generalisation
        modelAverageDistanceGeneralization, modelHausdorffDistanceGeneralization = generalisation(vae, testing_data)

        # calculate specificity
        modelSpecificity = specificity(decoder, testing_data, latent_dim, 20)

        # Save the results
        dataStore.append([i,
                          total_training_time,
                          modelAverageDistanceGeneralization,
                          modelHausdorffDistanceGeneralization,
                          modelSpecificity])

        # UI of progress
        print("The models training time is the following: {}".format(total_training_time))
        print("The models average distance generalisation is the following: {}".format(
            modelAverageDistanceGeneralization))
        print("The models Hausdorff distance generalisation is the following: {}".format(
            modelHausdorffDistanceGeneralization))
        print("The models specificity is the following: {}".format(modelSpecificity))
        print("--- Completed {} of {} folds ---".format(i + 1, k))

        K.clear_session()

    # Save the data to a csv
    header = ["Fold", "Time", "Avg.Generalisation", "Haus.Generalisation", "Specificity"]
    with open('results/' + dataName + modelName + trainingScheme + activations +'.csv', "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)  # write the header
        writer.writerows(dataStore)

    # ##################################################################################################################
    # Run a final time with all data to save weights for analysis later.
    # ##################################################################################################################

    # Works nicely for a single run
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    # go to terminal and run
    # tensorboard --logdir logs/
    original_dim = data.shape[1]

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=(original_dim,), name='encoder_input')

    # initiate the VAE model
    if (modelName == "DeepVAE"):
        encoder, z_mean, z_log_var = deep_encoder_model(inputs, latent_dim, intermediate_dim, "linear")
        # build decoder model
        decoder = deep_decoder_model(latent_dim, intermediate_dim, original_dim,
                                     activations,
                                     activations)

    else:  # Otherwise VAE
        encoder, z_mean, z_log_var = encoder_model(inputs, latent_dim)
        # build decoder model
        decoder = decoder_model(latent_dim, original_dim, activations)
        # instantiate VAE model

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name=dataName+modelName)
    print(encoder.summary())
    print(decoder.summary())
    print(vae.summary())

    # Creating the loss function MSE and KL divergence
    reconstruction_loss = mse(inputs, outputs)
    # reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=optimizers.adam(lr=learning_rate))
    vae.fit(data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data, None),
            callbacks=[tensorboard])
    vae.save('results/' + dataName + modelName + activations + '.h5')
    vae_json = vae.to_json()
    with open('results/' + dataName + modelName + activations + ".json", "w") as json_file:
        json_file.write(vae_json)
    # serialize weights to HDF5
    vae.save_weights('results/' + dataName + modelName + activations + '.h5')
    print("Saved model to disk")

def pointwiseDistanceTEST(dat1, dat2):

    # current dimension usually a long vector
    currentDim = dat1.shape[1]

    # get square difference an reshape
    sqrDiffDat = (dat1 - dat2)**2
    sqrDiffDat = np.reshape(sqrDiffDat, [int(currentDim/3), 3])

    # get point wise sums (row sum)
    pointwise_sqrDiffSum = np.sum(sqrDiffDat,axis =1)

    # Get pointwise euclidean distances
    pointwise_distances = np.sqrt(pointwise_sqrDiffSum)
    return pointwise_distances

def generalisationTEST(testingData,prediction):
    # Shouldnt be smaller than 0.002 as thats a GPMM for femurs for a kak VAE

    allAverageDistances = []
    allHausdorffDistances = []

    # Loop through all items in test data (LOOCV =1 but k>1)


    # item i
    current_mesh = np.reshape(testingData,[1,testingData.shape[0]])
    # models prediction of item i
    # _, _, z = encoder.predict(current_mesh)
    #
    # prediction_mesh = decoder.predict(z)

    prediction_mesh =np.reshape(prediction,[1,prediction.shape[0]])

    # get distance
    distance = pointwiseDistanceTEST(current_mesh, prediction_mesh)

    # get hausdorff
    hausdorff_distance = np.max(distance)
    allHausdorffDistances.append(hausdorff_distance)

    # get average
    average_distance = sum(distance)/(distance.size)
    allAverageDistances.append(average_distance)

    # for all training sets lets average over them
    # totalAverageDistances = sum(allAverageDistances) / len(allAverageDistances)
    # totalHausdorffDistances = sum(allHausdorffDistances) / len(allHausdorffDistances)
    print(average_distance, hausdorff_distance)


# def main():
#     import os
#     from  src.main.python.meshManipulation import loadMeshes,meshToData,meshToVec
#
#     os.chdir("/media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj2_python/")
#
#     # fetch data
#     currentMesh = loadMeshes("meshes/femurs/testCurrent/", ply_Bool=False)  # dims 50 36390
#     currentMeshData = np.array(meshToVec(currentMesh[0]))
#     meshProjection = loadMeshes("meshes/femurs/testProjected/",ply_Bool=False)
#
#     meshProjectionData = np.array(meshToVec(meshProjection[0]))
#
#     generalisationTEST(currentMeshData,meshProjectionData)
#
#
#
# if __name__ =="__main__":
#     main()
#
