# Script contains generalisation graphs for VAE results
# 16 December 2020
# Fabio Fehr

# FAUST WITH FAUST AND HAUS WITH HAUS

# plots
# FAUST - HAUS
# Femur - HAUS
# FAUST - AVG
# Femur - AVG

library(tidyverse)
# setwd("/media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj2_python/")
# getwd()

makeTidy = function(title, expName){
  dirName = paste0(title,expName)
  file = list.files(paste0("results/",
                           dirName,
                           "/modelMetrics/"))
  
  # fetch model metrics
  modelMetrics <- read_csv(paste0("results/",
                                  dirName,
                                  "/modelMetrics/",file),
                           col_names = T) %>% 
    as.data.frame() 
  
  # Create label 
  namedf <- data.frame(Name = rep(title,NROW(modelMetrics)))
  
  if(NCOL(modelMetrics) == 5){
    # return tidy stype data
    tidyModelMetrics <- modelMetrics %>% 
      cbind(.,namedf) %>% 
      gather("type", "Generalisation", -Time,-Fold,-Specificity, -Name) 
  } else{
    # return tidy stype data
    tidyModelMetrics <- modelMetrics %>% 
      cbind(.,namedf) %>% 
      gather("type", "Generalisation", -Time,-Fold,-Avg.Specificity,-Haus.Specificity, -Name) 
    
  }
  

}

####################################################################################################################################
# experiment 1
####################################################################################################################################


modelMetricsFemur <- makeTidy(title = "Femur","VAELOOCVtanh")
  
modelMetricsFAUST <- makeTidy("FAUST","VAELOOCVtanh")


modelMetricsTotal = rbind(modelMetricsFemur,modelMetricsFAUST)

modelMetricsTotal %>% 
  ggplot(aes(x = Name, y = Generalisation, fill =type))+
  stat_boxplot(geom = 'errorbar', linetype =1, width = 0.3, position= position_dodge(width = 0.7))+
  geom_boxplot(position = position_dodge(width = 0.7), width = 0.6)+
  scale_fill_manual(values =c("steelblue","goldenrod3"))+
  theme_light(base_size=15)+
  theme(legend.title = element_blank())+
  scale_y_continuous(breaks=seq(0,0.4,0.05), limits = c(0,0.4))+
  scale_x_discrete(name ="")+
  labs(title = "Baseline VAE", xlab =  element_blank())

# Save graph

ggsave(paste0("results/Exp1GeneralisationBoxplot.png"),
       width = 10, height = 6)

####################################################################################################################################
# experiment 2
####################################################################################################################################

# Femur (50,256), (50,512), (128,256), (128,512) (compare with vanilla)
# try HAUS and avg together and try them seperate
modelMetricsFemur_vanilla <- modelMetricsFemur %>%
  cbind(.,Experiment = "Baseline") %>%
  rename(Avg.Specificity= Specificity) %>% 
  cbind(.,Haus.Specificity = 0)

modelMetricsFemur_50_256 <- makeTidy("Femur", "DeepVAELOOCVlat50int256") %>% cbind(.,Experiment = "latent50hidden256")
modelMetricsFemur_50_512 <- makeTidy("Femur", "DeepVAELOOCVlat50int512") %>% cbind(.,Experiment = "latent50hidden512")
modelMetricsFemur_128_256 <- makeTidy("Femur", "DeepVAELOOCVlat128int256") %>% cbind(.,Experiment = "latent128hidden256")
modelMetricsFemur_128_512 <- makeTidy("Femur", "DeepVAELOOCVlat128int512") %>% cbind(.,Experiment = "latent128hidden512")

modelMetricsFemurTotal <- rbind(modelMetricsFemur_vanilla,
                                modelMetricsFemur_50_256,
                                modelMetricsFemur_50_512,
                                modelMetricsFemur_128_256,
                                modelMetricsFemur_128_512)

modelMetricsFemurTotal %>% filter(type == "Avg.Generalisation") %>% 
  ggplot(aes(x = Experiment, y = Generalisation, fill =type))+
  stat_boxplot(geom = 'errorbar', linetype =1, width = 0.3, position= position_dodge(width = 0.7))+
  geom_boxplot(position = position_dodge(width = 0.7), width = 0.6)+
  scale_fill_manual(values =c("steelblue","goldenrod3"))+
  theme_light(base_size=15)+
  theme(legend.position = "none")+
  scale_y_continuous(breaks=seq(0,0.6,0.005), limits = c(0,0.06))+
  scale_x_discrete(name ="")+
  labs(title = "Femur: Average generalisation", xlab =  element_blank())

ggsave(paste0("results/Exp2AvgGeneralisationBoxplotFemur.png"),
       width = 10, height = 7)

modelMetricsFemurTotal %>% filter(type == "Haus.Generalisation") %>% 
  ggplot(aes(x = Experiment, y = Generalisation, fill =type))+
  stat_boxplot(geom = 'errorbar', linetype =1, width = 0.3, position= position_dodge(width = 0.7))+
  geom_boxplot(position = position_dodge(width = 0.7), width = 0.6)+
  scale_fill_manual(values =c("goldenrod3"))+
  theme_light(base_size=15)+
  theme(legend.position = "none")+
  scale_y_continuous(breaks=seq(0,0.2,0.02), limits = c(0,0.2))+
  scale_x_discrete(name ="")+
  labs(title = "Femur: Hausdorff generalisation", xlab =  element_blank())

ggsave(paste0("results/Exp2HausGeneralisationBoxplotFemur.png"),
       width = 10, height = 7)

# Avg.Generalisation
FemurAvgGenExp2 <- modelMetricsFemurTotal %>% filter(type == "Avg.Generalisation")
aggregate(FemurAvgGenExp2$Generalisation,list(FemurAvgGenExp2$Experiment),mean)


# Haus.Generalisation
FemurHausGenExp2 <- modelMetricsFemurTotal %>% filter(type == "Haus.Generalisation")
aggregate(FemurHausGenExp2$Generalisation,list(FemurHausGenExp2$Experiment),mean)


# SPECIFICITY
aggregate(modelMetricsFemurTotal$Avg.Specificity,list(modelMetricsFemurTotal$Experiment),mean)

# TIME
aggregate(modelMetricsFemurTotal$Time,list(modelMetricsFemurTotal$Experiment),mean)

# Femur looks like vanilla is the best! 
# Lowest specificity, lowest times, lowest haus generalisation and lowest mean

# FAUST 

modelMetricsFAUST_vanilla <- modelMetricsFAUST %>%
  cbind(.,Experiment = "Baseline") %>%
  rename(Avg.Specificity= Specificity) %>% 
  cbind(.,Haus.Specificity = 0)

modelMetricsFAUST_100_256 <- makeTidy("FAUST", "DeepVAELOOCVlat100int256") %>% cbind(.,Experiment = "latent100hidden256")
modelMetricsFAUST_100_512 <- makeTidy("FAUST", "DeepVAELOOCVlat100int512") %>% cbind(.,Experiment = "latent100hidden512")
modelMetricsFAUST_128_256 <- makeTidy("FAUST", "DeepVAELOOCVlat128int256") %>% cbind(.,Experiment = "latent128hidden256")
modelMetricsFAUST_128_512 <- makeTidy("FAUST", "DeepVAELOOCVlat128int512") %>% cbind(.,Experiment = "latent128hidden512")
# modelMetricsFAUST_256_256 <- makeTidy("FAUST", "DeepVAELOOCVlat256int256") %>% cbind(.,Experiment = "latent256hidden256")

modelMetricsFAUSTTotal <- rbind(modelMetricsFAUST_vanilla,
                                modelMetricsFAUST_100_256,
                                modelMetricsFAUST_100_512,
                                modelMetricsFAUST_128_256,
                                # modelMetricsFAUST_256_256,
                                modelMetricsFAUST_128_512)

modelMetricsFAUSTTotal %>% filter(type == "Avg.Generalisation") %>% 
  ggplot(aes(x = Experiment, y = Generalisation, fill =type))+
  stat_boxplot(geom = 'errorbar', linetype =1, width = 0.3, position= position_dodge(width = 0.7))+
  geom_boxplot(position = position_dodge(width = 0.7), width = 0.6)+
  scale_fill_manual(values =c("steelblue","goldenrod3"))+
  theme_light(base_size=15)+
  theme(legend.position = "none")+
  scale_y_continuous(breaks=seq(0,0.12,0.01), limits = c(0,0.12))+
  scale_x_discrete(name ="")+
  labs(title = "FAUST: Average generalisation", xlab =  element_blank())

ggsave(paste0("results/Exp2AvgGeneralisationBoxplotFAUST.png"),
       width = 10, height = 7)

modelMetricsFAUSTTotal %>% filter(type == "Haus.Generalisation") %>% 
  ggplot(aes(x = Experiment, y = Generalisation, fill =type))+
  stat_boxplot(geom = 'errorbar', linetype =1, width = 0.3, position= position_dodge(width = 0.7))+
  geom_boxplot(position = position_dodge(width = 0.7), width = 0.6)+
  scale_fill_manual(values =c("goldenrod3"))+
  theme_light(base_size=15)+
  theme(legend.position = "none")+
  scale_y_continuous(breaks=seq(0,0.4,0.02), limits = c(0,0.4))+
  scale_x_discrete(name ="")+
  labs(title = "FAUST: Hausdorff generalisation", xlab =  element_blank())

ggsave(paste0("results/Exp2HausGeneralisationBoxplotFAUST.png"),
       width = 10, height = 7)

# Avg.Generalisation
FAUSTAvgGenExp2 <- modelMetricsFAUSTTotal %>% filter(type == "Avg.Generalisation")
aggregate(FAUSTAvgGenExp2$Generalisation,list(FAUSTAvgGenExp2$Experiment),mean)


# Haus.Generalisation
FAUSTHausGenExp2 <- modelMetricsFAUSTTotal %>% filter(type == "Haus.Generalisation")
aggregate(FAUSTHausGenExp2$Generalisation,list(FAUSTHausGenExp2$Experiment),mean)

# SPECIFICITY
aggregate(modelMetricsFAUSTTotal$Avg.Specificity,list(modelMetricsFAUSTTotal$Experiment),mean)

# TIME
aggregate(modelMetricsFAUSTTotal$Time,list(modelMetricsFAUSTTotal$Experiment),mean)

# Generlaisation FAUST - average - best is 100-512 then 100-256 and 128-512
# Generlaisation FAUST - Haus - best is 100-256 
# Specificity FAUST - average - best is 128-256 next close is 100 256
# time - best is vanilla obviously next best is 100 -256 

# THUS WE CHOOSE 100-256! 

# Save graph
# if(saveBoolean){
#   ggsave(paste0("results/",dirName,"/", dirName,"epochGraph.png"),
#          width = 10, height = 7)
# }


####################################################################################################################################
# experiment 3
####################################################################################################################################

# Femur
# 250 epochs
# intitialisations!
modelMetricsFemur <- makeTidy(title = "Femur","VAELOOCVtanh")
modelMetricsFemur_vanilla <- modelMetricsFemur %>%
  cbind(.,Experiment = "Baseline") %>%
  rename(Avg.Specificity= Specificity) %>% 
  cbind(.,Haus.Specificity = 0)

modelMetricsFemur_he_normal <- makeTidy("Femur", "VAELOOCVlat50int300he_normal") %>% cbind(.,Experiment = "heNormal")

modelMetricsFemur_glorot_uniform <- makeTidy("Femur", "VAELOOCVlat50int300glorot_uniform") %>% cbind(.,Experiment = "glorotUniform")


modelMetricsFemurinitialisationsTotal <- rbind(modelMetricsFemur_vanilla,
                                               modelMetricsFemur_he_normal,
                                               modelMetricsFemur_glorot_uniform)

modelMetricsFemurinitialisationsTotal %>% filter(type == "Avg.Generalisation") %>% 
  ggplot(aes(x = Experiment, y = Generalisation, fill =type))+
  stat_boxplot(geom = 'errorbar', linetype =1, width = 0.3, position= position_dodge(width = 0.7))+
  geom_boxplot(position = position_dodge(width = 0.7), width = 0.6)+
  scale_fill_manual(values =c("steelblue","goldenrod3"))+
  theme_light(base_size=15)+
  theme(legend.position = "none")+
  scale_y_continuous(breaks=seq(0,0.07,0.01), limits = c(0,0.07))+
  scale_x_discrete(name ="")+
  labs(title = "Femur: Average generalisation", xlab =  element_blank())

ggsave(paste0("results/Exp3AvgGeneralisationBoxplotFemur.png"),
       width = 10, height = 7)

modelMetricsFemurinitialisationsTotal %>% filter(type == "Haus.Generalisation") %>% 
  ggplot(aes(x = Experiment, y = Generalisation, fill =type))+
  stat_boxplot(geom = 'errorbar', linetype =1, width = 0.3, position= position_dodge(width = 0.7))+
  geom_boxplot(position = position_dodge(width = 0.7), width = 0.6)+
  scale_fill_manual(values =c("goldenrod3"))+
  theme_light(base_size=15)+
  theme(legend.position = "none")+
  scale_y_continuous(breaks=seq(0,0.18,0.02), limits = c(0,0.18))+
  scale_x_discrete(name ="")+
  labs(title = "Femur: Hausdorff generalisation", xlab =  element_blank())

ggsave(paste0("results/Exp3HausGeneralisationBoxplotFemur.png"),
       width = 10, height = 7)

# SPECIFICITY
aggregate(modelMetricsFemurinitialisationsTotal$Avg.Specificity,list(modelMetricsFemurinitialisationsTotal$Experiment),mean)

# TIME
aggregate(modelMetricsFemurinitialisationsTotal$Time,list(modelMetricsFemurinitialisationsTotal$Experiment),mean)

# Avg.Generalisation
FemurAvgGenExp3 <- modelMetricsFemurinitialisationsTotal %>% filter(type == "Avg.Generalisation")
aggregate(FemurAvgGenExp3$Generalisation,list(FemurAvgGenExp3$Experiment),mean)


# Haus.Generalisation
FemurHausGenExp3 <- modelMetricsFemurinitialisationsTotal %>% filter(type == "Haus.Generalisation")
aggregate(FemurHausGenExp3$Generalisation,list(FemurHausGenExp3$Experiment),mean)
# Average gen: He is much tighter, lowerquartiles, only 2 outliers (he is the best)
# Haus gen: fairly tight is he, but glorot uniform isnt bad either
# Time: He is the fastest (250 epochs) marginal 
# Glorot uniform has lowest specificity, but marginal to he. Both better than normal.

# Faust!
# 2000 epochs
# elu vs leaky relu
# he and glorot uniform

# vanilla
modelMetricsFAUST <- makeTidy("FAUST","VAELOOCVtanh")
modelMetricsFAUST_vanilla <- modelMetricsFAUST %>%
  cbind(.,Experiment = "Baseline") %>%
  rename(Avg.Specificity= Specificity) %>% 
  cbind(.,Haus.Specificity = 0)

# Best of the previous
modelMetricsFAUST_100_256 <- makeTidy("FAUST", "DeepVAELOOCVlat100int256") %>% cbind(.,Experiment = "latent100hidden256")

# New results
modelMetricsFAUST_elu_he_normal <- makeTidy("FAUST", "DeepVAELOOCVhe_normalelu") %>% cbind(.,Experiment = "eluHeNormal")
modelMetricsFAUST_elu_glorot_uniform <- makeTidy("FAUST", "DeepVAELOOCVglorot_uniformelu") %>% cbind(.,Experiment = "eluGlorotUniform")
modelMetricsFAUST_leaky_relu_he_normal <- makeTidy("FAUST", "DeepVAELOOCVhe_normalleakyRelu") %>% cbind(.,Experiment = "leakyReluHeNormal")
modelMetricsFAUST_leaky_relu_glorot_uniform <- makeTidy("FAUST", "DeepVAELOOCVglorot_uniformleakyRelu") %>% cbind(.,Experiment = "leakyReluGlorotUniform")

modelMetricsFAUSTinitialisationsEpochsActivationsTotal <- rbind(modelMetricsFAUST_vanilla,
                                                                modelMetricsFAUST_100_256,
                                                                 modelMetricsFAUST_elu_he_normal,
                                                                 modelMetricsFAUST_elu_glorot_uniform,
                                                                 modelMetricsFAUST_leaky_relu_he_normal,
                                                                 modelMetricsFAUST_leaky_relu_glorot_uniform)


modelMetricsFAUSTinitialisationsEpochsActivationsTotal %>% filter(type == "Avg.Generalisation") %>% 
  ggplot(aes(x = Experiment, y = Generalisation, fill =type))+
  stat_boxplot(geom = 'errorbar', linetype =1, width = 0.3, position= position_dodge(width = 0.7))+
  geom_boxplot(position = position_dodge(width = 0.7), width = 0.6)+
  scale_fill_manual(values =c("steelblue","goldenrod3"))+
  theme_light(base_size=15)+
  theme(legend.position = "none")+
  scale_y_continuous(breaks=seq(0,0.12,0.01), limits = c(0,0.12))+
  scale_x_discrete(name ="")+
  labs(title = "FAUST: Average generalisation", xlab =  element_blank())

ggsave(paste0("results/Exp3AvgGeneralisationBoxplotFAUST.png"),
       width = 14, height = 7)

modelMetricsFAUSTinitialisationsEpochsActivationsTotal %>% filter(type == "Haus.Generalisation") %>% 
  ggplot(aes(x = Experiment, y = Generalisation, fill =type))+
  stat_boxplot(geom = 'errorbar', linetype =1, width = 0.3, position= position_dodge(width = 0.7))+
  geom_boxplot(position = position_dodge(width = 0.7), width = 0.6)+
  scale_fill_manual(values =c("goldenrod3"))+
  theme_light(base_size=15)+
  theme(legend.position = "none")+
  scale_y_continuous(breaks=seq(0,0.4,0.02), limits = c(0,0.4))+
  scale_x_discrete(name ="")+
  labs(title = "FAUST: Hausdorff generalisation", xlab =  element_blank())

ggsave(paste0("results/Exp3HausGeneralisationBoxplotFAUST.png"),
       width = 14, height = 7)

# SPECIFICITY
aggregate(modelMetricsFAUSTinitialisationsEpochsActivationsTotal$Avg.Specificity,list(modelMetricsFAUSTinitialisationsEpochsActivationsTotal$Experiment),mean)

# TIME
aggregate(modelMetricsFAUSTinitialisationsEpochsActivationsTotal$Time,list(modelMetricsFAUSTinitialisationsEpochsActivationsTotal$Experiment),mean)

# Avg.Generalisation
FAUSTAvgGenExp3 <- modelMetricsFAUSTinitialisationsEpochsActivationsTotal %>% filter(type == "Avg.Generalisation")
aggregate(FAUSTAvgGenExp3$Generalisation,list(FAUSTAvgGenExp3$Experiment),mean)

# Haus.Generalisation
FAUSTHausGenExp3 <- modelMetricsFAUSTinitialisationsEpochsActivationsTotal %>% filter(type == "Haus.Generalisation")
aggregate(FAUSTHausGenExp3$Generalisation,list(FAUSTHausGenExp3$Experiment),mean)

# Average gen: at 2000 epochs, elu glorot was the best followed by leaky relu glorot uniform. 2000 epochs clearly made a difference
# Haus gen: elu glorot the best. leaky relu is the next best
# Time: elu glorot is the worst by far! leaky relu is the fastest of the 2000s obviously longer than the others
# Specificity: elu glorot is the worst then leaky relu the hes are better but naaa. 

# I think overall lets go for the best generalisation! So elu glorot (but leaky glorot is not bad either)

