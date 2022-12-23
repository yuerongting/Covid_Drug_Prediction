library(dplyr)
library(ggplot2)
library(OmnipathR)
library(igraph)
library(ggraph)
library(dbparser)
library(XML)
getwd()
setwd('c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation')




### Drug SMILES similarity

load("drug_name.Rdata")  ### All 468 drugs
drug_name
# load("drugs.Rdata")
# load("drug_salts.Rdata")

load("drug_interact_all.Rdata")
load("drug_calc_prop.Rdata")


### All SMILES 11,168
drug_SMILES <- (drug_calc_prop[which(drug_calc_prop[,1] == "SMILES"),]) [, c(4,2)]  
drug_SMILES_all <- drug_SMILES
# drug_SMILES_all <-filter(drug_SMILES, parent_key %in% unique(drug_interaction$`drugbank-id`))


# save(drug_SMILES_all, file = 'drug_SMILES.Rdata') # All SMILEs with DDIs

drug_SMILES_name <- unique(drug_name$`drugbank-id`)    ### 468 chemical drugs in DDI
# drug_SMILES<-drug_SMILES%>%filter(parent_key%in%drug_SMILES_name$`drugbank-id`) ### 258 out of 267 drugs have SMILES data
drug_SMILES <- drug_SMILES%>%filter(parent_key%in%drug_SMILES_name) ### All 468 Drug have SMILEs

# save(drug_SMILES, file='drug_SMILES_468.Rdata')  








load('drug_SMILES_468.Rdata')
drug_SMILES
drug_SMILES_468 = drug_SMILES

load('drug_SMILES.Rdata')
drug_SMILES_all ### All drug SMILES
drug_SMILES <- drug_SMILES_all


load('DDI_all.Rdata')
# load("DDI.Rdata")   ### 340  out of 468 drugs, have 48460 edges
DDI_node <- as_ids(V(DDI))







### SMILES similarity
load('DTI_graph.Rdata')
drug_target_inter_data

### All 3490 drugs have SMILES
drug_SMILES <- filter(drug_SMILES_all, parent_key %in% unique(union(DDI_node, drug_target_inter_data$parent_key))  ) ### 3490 (3366 + ...) drugs SMILES


SMILES_data <- as.matrix(drug_SMILES[,2])  ## Only SMILES vector

library(stringdist)
jaccard_sim <- stringsimmatrix(SMILES_data,SMILES_data, method = "jaccard")  ### Jaccard without padding (no SMILES)
rownames(jaccard_sim) <- drug_SMILES$parent_key
colnames(jaccard_sim) <- drug_SMILES$parent_key


example <- c("CC(N)CC1=CC=CC=C1", "CN1CCC[C@H]1C1=CN=CC=C1")
Reduce(intersect, example)

example_1 <-as.matrix(data.frame(y1 = c("CC(N)CC1=CC=CC=C1")))
example_2 <-as.matrix(data.frame(y1 = c("CN1CCC[C@H]1C1=CN=CC=C1"))) # 0.4
# example_2 <-as.matrix(data.frame(y1 = c("CC(C)(N)CC1=CC=CC=C1"))) # 1
# intersect(example_1, example_2)

example <- as.matrix(data.frame(y1 = c("CC(N)CC1=CC=CC=C1", "CN1CCC[C@H]1C1=CN=CC=C1")))
stringsimmatrix(example,example, method = "jaccard", q=2)

# 
# x <- "CC(N)CC1=CC=CC=C1"
# y <- "CN1CCC[C@H]1C1=CN=CC=C1"
x_q = qgrams(A = example_1,  q=1)
y_q = qgrams(B = example_2, q=1)

a = example
b = example

# library(udpipe)
# txt_overlap(x_q, y_q)
# 
# intersect(x_q, y_q)
# 
# install.packages('quanteda')
# library(quanteda)
stringdist::stringdistmatrix(example_1, example_2, method = "qgram", 
                             useBytes = TRUE, q = 1)

stringdist::stringdistmatrix(example_2, example_1, method = "jaccard", 
                             useBytes = TRUE, q = 1)


### Drugs with no SMILE data
# add_name <- setdiff(drug_common, rownames(jaccard_sim))   ### common drugs that not have SMILES
# add_zero_col <- matrix(0, nrow = nrow(jaccard_sim) , ncol = length(add_name)) 
# add_zero_row <- matrix(0, nrow = length(add_name), ncol = ncol(jaccard_sim)+dim(add_zero_col)[2]) 
# add_zero_row <- matrix(0, nrow = length(add_name), ncol = ncol(jaccard_sim)) 
# jaccard_sim <- cbind(jaccard_sim, add_zero_col)
# jaccard_sim <- rbind(jaccard_sim, add_zero_row)

# rownames(jaccard_sim) <- c(drug_SMILES$parent_key,add_name)
# colnames(jaccard_sim) <- c(drug_SMILES$parent_key,add_name)

# intersect(drug_SMILES$parent_key,row.names(jaccard_sim) )
# intersect(rownames(jaccard_sim), drug_common)
# intersect(rownames(DTI), drug_common)

load('drug_common.Rdata')
drug_common  ### 214 drugs from DTI


### Column only drugs in DTI
# jaccard_sim <- jaccard_sim[, colnames(jaccard_sim) %in% drug_common]  ### 3490 x 214
dim(jaccard_sim)  

### 214 drugs DTI
jaccard_sim_DTI_drugs <- jaccard_sim[rownames(jaccard_sim) %in% drug_common,] ### 214 drugs
dim(jaccard_sim_DTI_drugs)
jaccard_sim_DTI_drugs <- jaccard_sim_DTI_drugs[order(rownames(jaccard_sim_DTI_drugs)) , order(colnames(jaccard_sim_DTI_drugs))]
jaccard_sim_DTI_drugs <- as.data.frame(jaccard_sim_DTI_drugs)
dim(jaccard_sim_DTI_drugs)  


### 468 Drugs in  DDI
jaccard_sim_468_DDI <- jaccard_sim[rownames(jaccard_sim) %in% drug_SMILES_468$parent_key,] 
# jaccard_sim_468_DDI <- jaccard_sim_468_DDI[rownames(jaccard_sim_468_DDI) %!in% drug_common,] 
jaccard_sim_468_DDI <- jaccard_sim_468_DDI[order(rownames(jaccard_sim_468_DDI)) , order(colnames(jaccard_sim_DTI_drugs))]
jaccard_sim_468_DDI <- as.data.frame(jaccard_sim_468_DDI)
dim(jaccard_sim_468_DDI)   ### 468 x 214

### Other drugs
`%!in%` <- Negate(`%in%`)
jaccard_sim_other <- jaccard_sim[rownames(jaccard_sim) %!in% union(drug_common, drug_SMILES_468$parent_key),] ### 266 drugs verse 30 drug of interest
jaccard_sim_other <- jaccard_sim_other[order(rownames(jaccard_sim_other)) , order(colnames(jaccard_sim_other))]
jaccard_sim_other <- as.data.frame(jaccard_sim_other)
dim(jaccard_sim_other)    ### 3022 x 214

# load("DTI.Rdata")
# DTI

# intersect(row.names(jaccard_sim) , rownames(DTI))
# intersect(drug_common, drug_SMILES_name)
# intersect(intersect(drug_common, drug_SMILES_name), rownames(DTI))

# save(jaccard_sim_DTI_drugs, file = "jaccard_sim_DTI_drug.Rdata")  ### 214 to 214 DTI drugs
# save(jaccard_sim_468_DDI, file = "jaccard_sim_label_0.Rdata")  ### 468 DDI drugs (training, with labels)
# save(jaccard_sim_other, file = "jaccard_sim_all_other_to_all.Rdata")  ### Other drugs (testing)





# save(jaccard_sim_468_DDI, file = "jaccard_sim_468_DDI.Rdata")
# 