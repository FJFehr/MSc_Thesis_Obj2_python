# Script contains epoch graphs for VAE results
# 15 December 2020
# Fabio Fehr

library(tidyverse)
setwd("/media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj2_python/")

# getwd()

# average errors over all 100, 50 observations per epoch
# validation error is the error on the left out observations
# training error is the error on all 99, 49 observations

# USING means and CI
epochGraph <- function(title, expName, plotTitle, lowerBound,upperBound,saveBoolean, totalEpochs=1000){
  
  dirName = paste0(title,expName)
  print(paste("Currently viewing ",title, expName))
  
  epochsdf <- data.frame(Epochs = 1:totalEpochs) 
  
  trainingfile = list.files(paste0("results/",
                                   dirName,
                                   "/trainingLoss/"))
  
  testingfile = list.files(paste0("results/",
                                  dirName,
                                  "/testingLoss/"))
  
  # fetch training epochs + clean
  trainingEpochs <- read_csv(paste0("results/",
                                    dirName,
                                    "/trainingLoss/",trainingfile),
                             col_names = F) %>% 
    t() %>%                                 # make epochs rows and observations columns
    as.data.frame() 
  
  
  # fetch testing epochs + clean
  testingEpochs <- read_csv(paste0("results/",
                                   dirName,
                                   "/testingLoss/",testingfile),
                            col_names = F) %>% 
    t() %>%                                 # make epochs rows and observations columns
    as.data.frame() 
  
  n = NCOL(testingEpochs) # observations
  nEpochs = NROW(testingEpochs) # number of epochs
  
  
  # Get means
  trainingrow_mean = apply(trainingEpochs,1,mean)
  testingrow_mean = apply(testingEpochs,1,mean)
  
  # Get sd
  trainingrow_sd = apply(trainingEpochs,1,sd)
  testingrow_sd = apply(testingEpochs,1,sd)
  
  # Get upper CI
  trainingupper = trainingrow_mean + 1.96 * (trainingrow_sd / sqrt(n))
  testingupper = testingrow_mean + 1.96 * (testingrow_sd / sqrt(n))
  
  # Get lower CI
  traininglower = trainingrow_mean - 1.96 * (trainingrow_sd / sqrt(n))
  testinglower = testingrow_mean - 1.96 * (testingrow_sd / sqrt(n))
  
  # Make dfs
  trainingdf <- data.frame(Epochs = 1:nEpochs,
                           Loss = trainingrow_mean,
                           upper = trainingupper,
                           lower = traininglower,
                           name = "Training Loss")
  
  testingdf <- data.frame(Epochs = 1:nEpochs,
                          Loss = testingrow_mean,
                          upper = testingupper,
                          lower = testinglower,
                          name = "Validation Loss")
  
  colors <- c("Training Loss" = "dodgerblue", "Validation Loss" = "grey40")
  
  ggplot(trainingdf, aes(Epochs, Loss))+
    geom_line(aes(colour = "Training Loss"))+
    ylim(lowerBound,upperBound)+
    geom_ribbon(aes(ymin=lower,ymax=upper),fill = "dodgerblue",alpha=0.4)+
    geom_line(data =testingdf, aes(Epochs, Loss, colour = "Validation Loss"))+
    geom_ribbon(data =testingdf, aes(ymin=lower,ymax=upper),fill = "grey40",alpha=0.4)+
    theme_light()+
    theme(legend.title = element_blank(),text = element_text(size = 20))+
    scale_color_manual(values = colors)+
    labs(title = plotTitle)
  
  # Save graph
  if(saveBoolean){
    ggsave(paste0("results/",dirName,"/", dirName,"epochGraph.png"),
      width = 10, height = 7)
  }

}

# Experiment1 Vanilla VAE #################################################################################################
epochGraph("FAUST","VAELOOCVtanh","FAUST: latent100",50,100,saveBoolean=T)
epochGraph("Femur","VAELOOCVtanh","Femur: latent50",0,50,saveBoolean=T)

# training continues to decrease

# validation plateaus (justifies 1000 is enough epochs)

# Experiment 2: Deep models #################################################################################################

epochGraph("FAUST","DeepVAELOOCVlat100int256","FAUST: latent100hidden256",20,100,saveBoolean=T)
# Latent dimension means and sd =  100
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 300 epochs 
# intersect: at 270 epochs

epochGraph("Femur","DeepVAELOOCVlat50int256","Femur: latent50hidden256",0,50,saveBoolean=T)
# Latent dimension means and sd =  50
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (gets marginally smaller) 
# intersect: no interesction!

epochGraph("FAUST","DeepVAELOOCVlat100int512","FAUST: latent100hidden512",20,100,saveBoolean=T)
# Latent dimension means and sd =  100
# intermediate dimension number of nodes = 512
# training: continues to decrease
# validation: plateaus at 500 epochs 
# intersect: at 500 epochs

epochGraph("Femur","DeepVAELOOCVlat50int512","Femur: latent50hidden512",0,50,saveBoolean=T)
# Latent dimension means and sd =  50
# intermediate dimension number of nodes = 512
# training: continues to decrease
# validation: plateaus at 1000 epochs (loss = 17.5)
# intersect: no intersection

epochGraph("FAUST","DeepVAELOOCVlat128int256","FAUST: latent128hidden256",20,100,saveBoolean=T)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 300 epochs (loss = 65)
# intersect: at 300

epochGraph("Femur","DeepVAELOOCVlat128int256","Femur: latent128hidden256",0,50,saveBoolean=T)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (loss = 16)
# intersect: no intersection

epochGraph("FAUST","DeepVAELOOCVlat256int256","FAUST: latent256hidden256",20,100,saveBoolean=T)
# Latent dimension means and sd =  256
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (loss = 65)
# intersect: at 350 epochs

#####################

epochGraph("FAUST","DeepVAELOOCVlat128int512","FAUST: latent128hidden512",20,100,saveBoolean=T)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 300 epochs (loss = 65)
# intersect: at 300

epochGraph("Femur","DeepVAELOOCVlat128int512","Femur: latent128hidden512",0,50,saveBoolean=T)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (loss = 16)
# intersect: no intersection

##########################################################################################################################

epochGraph("Femur","VAELOOCVlat50int300he_normal","Femur: heNormal",0,50,saveBoolean=T)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (loss = 16)
# intersect: no intersection

epochGraph("Femur","VAELOOCVlat50int300glorot_uniform","Femur: glorotUniform",0,50,saveBoolean=T)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (loss = 16)
# intersect: no intersection

####### FAUST ##############################

epochGraph("FAUST","DeepVAELOOCVhe_normalelu","FAUST: eluHeNormal",20,200,saveBoolean=T,totalEpochs = 2000)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (loss = 16)
# intersect: no intersection

epochGraph("FAUST","DeepVAELOOCVglorot_uniformelu","FAUST: eluGlorotUniform",20,100,saveBoolean=T,totalEpochs = 2000)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (loss = 16)
# intersect: no intersection

epochGraph("FAUST","DeepVAELOOCVhe_normalleakyRelu","FAUST: leakyReluHeNormal",20,200,saveBoolean=T,totalEpochs = 2000)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (loss = 16)
# intersect: no intersection

epochGraph("FAUST","DeepVAELOOCVglorot_uniformleakyRelu","FAUST: leakyReluGlorotUniform",20,100,saveBoolean=T,totalEpochs = 2000)
# Latent dimension means and sd =  128
# intermediate dimension number of nodes = 256
# training: continues to decrease
# validation: plateaus at 500 epochs (loss = 16)
# intersect: no intersection


# VALIDATION LOSS < TRAINING LOSS. Training has the regulaisation loss with it, validation set is small thus easy
# https://www.pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/

# experiments #######################################################################################################

# USING SMOOTHS
# epochGraph <- function(title, expName, lowerBound,upperBound,saveBoolean){
#   
#   dirName = paste0(title,expName)
#   print(paste("Currently viewing ",title, expName))
#   
#   epochsdf <- data.frame(Epochs = 1:1000) 
#   testNamedf <- data.frame(Name = rep("Validation error",100000))
#   trainNamedf <- data.frame(Name = rep("Training error",100000))
#   
#   trainingfile = list.files(paste0("results/",
#                                    dirName,
#                                    "/trainingLoss/"))
#   
#   testingfile = list.files(paste0("results/",
#                                   dirName,
#                                   "/testingLoss/"))
#   
#   # fetch training epochs + clean
#   trainingEpochs <- read_csv(paste0("results/",
#                                     dirName,
#                                     "/trainingLoss/",trainingfile),
#                              col_names = F) %>% 
#     t() %>%                                 # make epochs rows
#     as.data.frame() %>% 
#     cbind(epochsdf,.) %>%                   # Label the epochs
#     gather("run", "Loss",-Epochs) %>%       # make tidy data
#     cbind(trainNamedf,.)                    # Label entire dataset
#   
#   trainEpoch1000Loss = trainingEpochs %>% filter(Epochs == 1000) %>% .$Loss %>% mean()
#   print(paste("The training final loss is: ",trainEpoch1000Loss))
#   
#   # fetch testing epochs + clean
#   testingEpochs <- read_csv(paste0("results/",
#                             dirName,
#                             "/testingLoss/",testingfile),
#                             col_names = F) %>% 
#     t() %>%                                 # make epochs rows
#     as.data.frame() %>% 
#     cbind(epochsdf,.) %>%                   # Label the epochs
#     gather("run", "Loss",-Epochs) %>%       # make tidy data
#     cbind(testNamedf,.)                     # Label entire dataset
#   
#   print(testingEpochs %>% filter(Epochs == 1000))
#   testEpoch1000Loss = testingEpochs %>% filter(Epochs == 1000) %>% .$Loss %>% mean()
#   print(paste("The validation final loss is: ",testEpoch1000Loss))
#   
#   
#   # Create graph
#   rbind(trainingEpochs,testingEpochs) %>% 
#     ggplot(aes(x=Epochs,y=Loss))+
#     ylim(lowerBound,upperBound)+
#     geom_smooth(aes(col = Name))+
#     scale_color_manual(values=c("grey60","dodgerblue"))+
#     labs(title = title)+
#     theme_light()+
#     theme(legend.title = element_blank())
#   
#   # Save graph
#   # if(saveBoolean){
#   #   ggsave(paste0("results/",dirName,"/", dirName,"epochGraph.png"),
#   #     width = 10, height = 7)
#   # }
#   
# }

# Smoothing the lines

# newdf <- apply(dftesting,2,smooth.spline)
# smoothdf <- data.frame(epochs = 1:1000,
#                        mean = newdf$mean$y,
#                        upper = newdf$upper$y,
#                        lower = newdf$lower$y)
# 
# newdf1 <- apply(dftraining,2,smooth.spline)
# smoothdf1 <- data.frame(epochs = 1:1000,
#                        mean = newdf1$mean$y,
#                        upper = newdf1$upper$y,
#                        lower = newdf1$lower$y)
# 
# ggplot(smoothdf1, aes(epochs, mean))+
#   geom_line(col = "dodgerblue",size=1)+
#   ylim(40,150)+
#   geom_ribbon(aes(ymin=lower,ymax=upper),fill = "dodgerblue",alpha=0.4)+
#   geom_line(data =smoothdf, aes(epochs, mean), col = "grey40",size=1)+
#   geom_ribbon(data =smoothdf, aes(ymin=lower,ymax=upper),fill = "grey40",alpha=0.4)+
#   theme_light()




