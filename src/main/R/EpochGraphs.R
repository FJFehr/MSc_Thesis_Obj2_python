# Script contains epoch graphs for VAE results
# 15 December 2020
# Fabio Fehr

# average errors over all 100, 50 observations per epoch
# validation error is the error on the left out observations
# training error is the error on all 99, 49 observations

# training continues to decrease
# validation plateaus (justifies 1000 is enough epochs)

library(tidyverse)
setwd("../../media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj2_python/")
# getwd()

epochGraph <- function(title, expName, lowerBound,upperBound,saveBoolean){
  
  dirName = paste0(title,expName)
  
  epochsdf <- data.frame(Epochs = 1:1000) 
  testNamedf <- data.frame(Name = rep("Validation error",100000))
  trainNamedf <- data.frame(Name = rep("Training error",100000))
  
  trainingEpochs <- read_csv(paste0("results/",
                                    dirName,
                                    "/trainingLoss/trainingLoss",dirName,".csv"),
                             col_names = F) %>% 
    t() %>%                                 # make epochs rows
    as.data.frame() %>% 
    cbind(epochsdf,.) %>%                   # Label the epochs
    gather("run", "Loss",-Epochs) %>%       # make tidy data
    cbind(trainNamedf,.)                    # Label entire dataset
  
  testingEpochs <- read_csv(paste0("results/",
                            dirName,
                            "/testingLoss/testingLoss",dirName,".csv"),
                            col_names = F) %>% 
    t() %>%                                 # make epochs rows
    as.data.frame() %>% 
    cbind(epochsdf,.) %>%                   # Label the epochs
    gather("run", "Loss",-Epochs) %>%       # make tidy data
    cbind(testNamedf,.)                     # Label entire dataset
  
  rbind(trainingEpochs,testingEpochs) %>% 
    ggplot(aes(x=Epochs,y=Loss))+
    ylim(lowerBound,upperBound)+
    geom_smooth(aes(col = Name))+
    scale_color_manual(values=c("grey60","dodgerblue"))+
    labs(title = title)+
    theme_light()+
    theme(legend.title = element_blank())
  
  if(saveBoolean){
    ggsave(paste0("results/",dirName,"/", dirName,"epochGraph.png"),
      width = 10, height = 7)
  }
  
}

epochGraph("FAUST","VAELOOCVtanh",50,100,saveBoolean=T)
epochGraph("Femur","VAELOOCVtanh",0,40,saveBoolean=T)
