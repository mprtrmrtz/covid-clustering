
rm(list = ls())


library(tidyverse)
library(stargazer)


df <- read.csv('COVID_Clustering_data_regression.csv')

df$HE.VS <- NULL
df$HE.CS <- NULL
df$X <- NULL

data_cluster_1 <-  filter(df, KMeans_Labels == 1)
data_cluster_2 <-  filter(df, KMeans_Labels == 2)
data_cluster_3 <-  filter(df, KMeans_Labels == 3)
data_cluster_4 <-  filter(df, KMeans_Labels == 4)
data_cluster_5 <-  filter(df, KMeans_Labels == 5)



reg1 <- lm(data = data_cluster_1, data_cluster_1$HE.VH ~ . - KMeans_Labels)
summary(reg1)

reg2 <- lm(data = data_cluster_2, data_cluster_2$HE.VH ~ . - KMeans_Labels)
summary(reg2)


reg3 <- lm(data = data_cluster_3, data_cluster_3$HE.VH ~ . - KMeans_Labels)
summary(reg3)


reg4 <- lm(data = data_cluster_4, data_cluster_4$HE.VH ~ . - KMeans_Labels)
summary(reg4)


reg5 <- lm(data = data_cluster_5, data_cluster_5$HE.VH ~ . - KMeans_Labels)
summary(reg5)


# Load the necessary library
library(stargazer)

# Assuming you have your models stored in model1, model2, model3, model4, and model5

# Define the custom variable names
variable_labels <- c("MMR Vaccination Coverage (0-1)", "Population 2020", "Cumulative Death Rate (Dec 2021)", 
                     "Black Population (%)", "Hispanic Population (%)", "Higher Education (%)", 
                     "Median Household Income", "Median Age", "Vehicles per Household", 
                     "Without Insurance (%)", "Republican (%)", "Constant")

# Generate the stargazer table
stargazer(reg1, reg2, reg3, reg4, reg5,
          title = "Regression Results for Vaccine Hesitancy (Dec 2021)",
          # dep.var.labels = c("Vaccine Hesitancy (Dec 2021)"),
         # covariate.labels = variable_labels,
          type = "text",
          keep.stat = c("n", "rsq", "adj.rsq"),
          column.sep.width = "5pt",
          digits = 3,
          font.size = "small",
          notes = "Note: *p<0.1; **p<0.05; ***p<0.01",
          label = "tab:vaccine_hesitancy",
          header = FALSE)






