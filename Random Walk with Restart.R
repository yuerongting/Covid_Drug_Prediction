library(RandomWalkRestartMH)
library(igraph)
library(RCy3)
library(STRINGdb)
library(string_db)
library(influential)
library(dplyr)
library(ggplot2)
# library(OmnipathR)
library(igraph)
library(ggraph)
# data(PPI_Network)
# data(Pathway_Network) # We load the Pathway Network
# ID <- c("FN1","HAMP","ILK","MIF","NME1","PROCR","RAC1","RBBP7",
#         "TMEM176A","TUBG1","UBC","VKORC1")

getwd()
setwd('C:\\Users\\Yuerongting\\Desktop\\Covid_Project\\Covid_Project\\Covid_coference_presentation')  # \\ abs path

##### new DEG list
PPI_Network_1 = sif2igraph("string_interactions_short.tsv.sif", directed = FALSE)

DTI = sif2igraph("stitch_interactions.sif", directed = FALSE)
# plot(DTI, vertex.size=6)  
###### Random work with restart

# ## PPI
# PPI_PATH_Multiplex <- 
#   create.multiplex(list(PPI=PPI_Network_1))
# ## DTI
# DTI_Multiplex <- 
#   create.multiplex(list(PPI=DTI))

# PPI + DTI
MultiplexObject <- create.multiplex(list(PPI=PPI_Network_1, DTI = DTI))

AdjMatrix_PPI <- compute.adjacency.matrix(MultiplexObject)
AdjMatrixNorm_PPI <- normalize.multiplex.adjacency(AdjMatrix_PPI)

SeedGene <- c("TLR2","ITGB3","CFP","PIK3CA","BAX","CCL5","IL6","IL1B") #"HLA-DMB","PILRB","ZFP37",
# SeedGene <- c("ICAM1") # ICAM1, ITGA2, FGF2, SERPINE1
## We launch the algorithm with the default parameters (See details on manual)
RWR_PPI_Results <- Random.Walk.Restart.Multiplex(AdjMatrixNorm_PPI,
                                                 MultiplexObject,SeedGene)


save(RWR_PPI_Results, file = "RWR_PPI_Results.Rdata")


for(i in drug_name){
  if (rownames(RWR_PPI_Results) == i){
    
  }
}





TopResults_PPI <-
  create.multiplexNetwork.topResults(RWR_PPI_Results,MultiplexObject,
                                     k=150)
par(mar=c(0.1,0.1,0.1,0.1))
plot(TopResults_PPI, vertex.label.color="black",vertex.frame.color="#ffffff",
     vertex.size= 6, edge.curved=.2,
     vertex.color = ifelse(igraph::V(TopResults_PPI)$name == SeedGene,"red",
                           "#00CCFF"), edge.color="blue",edge.width=0.8)
RWR_PPI_Results



