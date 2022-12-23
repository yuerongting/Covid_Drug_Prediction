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

getwd()
setwd('c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation')

load("DDI_all.Rdata")  ### All DDI, 3366 drugs
DDI
drug_DDI_all <- as_ids(V(DDI))

load("DDI.Rdata")   ### 340 drugs
# load("DDI_258_drugs.Rdata")  ### 281 drugs
# DDI
DDI_dat_frame <- as.data.frame(get.edgelist(DDI))
# save(DDI_dat_frame, file = 'DDI_dat_frame.Rdata') # DTI for python, 340 drugs


vertex_DDI <- V(DDI) 

PPI_Network_1 = sif2igraph("string_interactions_short.tsv.sif", directed = FALSE)






load("drug_target_inter_data.Rdata")
drug_target_inter_data
unique(drug_target_inter_data$parent_key)  ### 214 drugs in DTI

###### Save DTI
DTI_dat_frame <- as.data.frame(drug_target_inter_data)
# save(DTI_dat_frame, file = 'DTI_dat_frame.Rdata')
# write.csv(DTI_dat_frame,'c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation\\DTI_dat_frame.csv')


load("drug_SMILES.Rdata")
drug_SMILES_all


DTI_graph <- filter(drug_target_inter_data, parent_key %in% drug_SMILES_all$parent_key ) # 16 out 30 are in DDI (have SMILES)


DTI = graph_from_data_frame(drug_target_inter_data, directed = FALSE)   ### 295 DTI edges



load("gene_name.Rdata") 
gene_name   # 495 proteins


target_node_DTI <- intersect(as_ids(V(DTI)), gene_name)

# Node_diff <- c(  setdiff(   unique(as_ids(V(DDI) )) , unique(DTI_dat_frame$parent_key)  )  )  ### 214 drugs, Add 254 diff nodes

Node_diff <- c(  setdiff(   unique(as_ids(V(DDI) )) , unique(DTI_dat_frame$parent_key)  )  )  ### 214 drugs (unique), Add 254 diff nodes

# `%!in%` <- Negate(`%in%`)
# save(Node_diff, file = 'Node_diff.Rdata')


DTI_graph <- DTI + vertices( Node_diff )  ### add 254 diff drug nodes
# save(DTI_graph, file = 'DTI_graph.Rdata')



load("drug_name.Rdata")  ### DDI 468 drugs
drug_name

DDI <- DDI + vertices( setdiff(drug_name$`drugbank-id`, as_ids( V(DDI)))    )   ### DDI 468 drugs

# plot(DDI)
# unique(V(DTI_graph))

PPI_Network_1
PPI_Network_1 <- delete_vertices(PPI_Network_1, setdiff(  as_ids(V(PPI_Network_1) ) , gene_name )   )
load("protein_seq_similarity.RData")
node_name <- rownames(protein_seq_similarity)
Node_diff <- unique(c(setdiff( unique(node_name), unique(as_ids(V(PPI_Network_1) ))  )))  ### 212 proteins, Add 287 diff nodes
PPI_Network_1 <- PPI_Network_1 + vertices( Node_diff)



###### Save PPI
# PPI_dat_frame <- as.data.frame(get.edgelist(PPI_Network_1))
# PPI_dat_frame <- as.data.frame( as_ids(V(PPI_Network_1)) )
# colnames(PPI_dat_frame) <- 'prot'
# PPI_dat_frame <- as.data.frame(  PPI_dat_frame[order(PPI_dat_frame$prot),1]  )
# 
# colnames(PPI_dat_frame) <- 'prot'
# # save(PPI_dat_frame, file = 'PPI_dat_frame.Rdata')
# # write.csv(PPI_dat_frame,'c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation\\PPI_dat_frame.csv')
# 
# head(PPI_dat_frame)

###### Save DDI
# DDI_dat_frame <- as.data.frame(get.edgelist(DDI))
# # DDI_dat_frame <- as.data.frame( as_ids(V(DDI)) )
# colnames(DDI_dat_frame) <- c('drug_v1', 'drug_v2')
# # DDI_dat_frame <- as.data.frame(  DDI_dat_frame[order(DDI_dat_frame$drug_v1),1]  )
# 
# # save(PPI_dat_frame, file = 'PPI_dat_frame.Rdata')
# # write.csv(PPI_dat_frame,'c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation\\PPI_dat_frame.csv')
# 
# head(DDI_dat_frame)





### Multiplex and Heterogeneous
library('dplyr')


Multiplex_DDI <- create.multiplex(list(DDI = DDI))  
Multiplex_PPI <- create.multiplex(list(PPI = PPI_Network_1))


# DTI_data_frame <- as.data.frame(drug_target_inter_data)
# DTI_data_frame <- as.data.frame(DTI_graph)
DTI_data_frame <- as.data.frame(get.edgelist(DTI_graph), directed = FALSE)

# save(Multiplex_DDI, file = 'Multiplex_DDI.Rdata')  ### Saved 
# save(Multiplex_PPI, file = 'Multiplex_PPI.Rdata')
# save(DTI_data_frame, file = 'DTI_data_frame.Rdata')


### Check
# unique(as_ids(V(DDI)) )  ### 3366 drugs
# unique(as_ids(V(DDI)) )  ### 340 drugs
unique(as_ids(V(DDI)) )  ### 468 drugs
unique(drug_target_inter_data$parent_key)  ### 214 drugs


setdiff(unique(as_ids(V(DDI)) ), unique(drug_target_inter_data$parent_key) )   ### 254 drugs
setdiff(unique(drug_target_inter_data$parent_key), unique(as_ids(V(DDI)) ) )   ### 128 drugs

setdiff(unique(as_ids(V(DDI)) ), unique(DTI_data_frame$V1) )
# setdiff(DTI_data_frame$from, as_ids( V(DDI)  )  )






### Heterogenous network

DTI_data_frame <- filter(DTI_data_frame, V1 %in% as_ids(V(DDI))  )
# multiplexHet <- create.multiplexHet(test, Multiplex_PPI, DTI_data_frame)

ndoes <- unique(DTI_data_frame$V1)    ##  86 drug nodes in common in DTI/DDI
 

DTI_data_frame$V1








### Edge of all 3 graphs
DDI_edge <- as.data.frame(get.edgelist(DDI))
PPI_edge <- as.data.frame(get.edgelist(PPI_Network_1))
DTI_edge <- DTI_data_frame

# all_edge_type = c("DDI", "PPI", "DTI")
all_edge_type = c(1, 2, 3)
DDI_label = all_edge_type[1]
DDI_edge <- data.frame(append(DDI_edge, c(edge_type=DDI_label), after=2)) # DDI label

PPI_label = all_edge_type[2]
PPI_edge <- data.frame(append(PPI_edge, c(edge_type=PPI_label), after=2)) # PPI label

DTI_label = all_edge_type[3]
DTI_edge <- data.frame(append(DTI_edge, c(edge_type=DTI_label), after=2)) # DTI label


edge_all <- bind_rows(DDI_edge, bind_rows(PPI_edge, DTI_edge))
# write.csv(edge_all,'c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation\\edge_all_label.csv')
###






colnames(DDI_dat_frame) <- c('drug_v1', 'drug_v2')
# DDI_dat_frame <- as.data.frame(  DDI_dat_frame[order(DDI_dat_frame$drug_v1),1]  )

# save(PPI_dat_frame, file = 'PPI_dat_frame.Rdata')
# write.csv(PPI_dat_frame,'c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation\\PPI_dat_frame.csv')

head(DDI_dat_frame)


multiplexHet <- create.multiplexHet(Multiplex_DDI, Multiplex_PPI, DTI_data_frame)


# save(Multiplex_DDI, file = "Multiplex_DDI.Rdata")
# save(multiplexHet, file = "multiplexHet.Rdata")



# plot(multiplexHet)
MultiHetTranMatrix <- compute.transition.matrix(multiplexHet)
MultiHetTranMatrix@x[is.nan(MultiHetTranMatrix@x)] <- 0
dim(MultiHetTranMatrix) 
head(MultiHetTranMatrix)
# intersect(as_ids(V(DDI)), as_ids((V(DTI))))





######### adj matrix of all graph
adj_all <- as.data.frame(as.matrix(multiplexHet$BipartiteNetwork))   ### adj = drug * target
rownames(adj_all)<-gsub("_1","",rownames(adj_all))
colnames(adj_all)<-gsub("_1","",colnames(adj_all))

# save(adj_all, file = 'adj_all.Rdata')

#########




############### Edge_weight (295)
transit_mat <- as.matrix(MultiHetTranMatrix) ### transition matrix
rownames(transit_mat)<-gsub("_1","",rownames(transit_mat))
colnames(transit_mat)<-gsub("_1","",colnames(transit_mat))
transit_mat <- transit_mat[rownames(transit_mat) %in% gene_name,]  ### row protein
transit_mat <- transit_mat[,colnames(transit_mat) %in% drug_name$`drugbank-id`]   ### column drug

sum(transit_mat[,7])

ind <- which(transit_mat!=0,  arr.ind = TRUE)  ### nonzero elements 295 == number of interactions

link_row <- rownames(transit_mat)[ind[,1]]
link_col <- colnames(transit_mat)[ind[,2]]

edge_weight <- as.data.frame(transit_mat[ind])

# save(edge_weight, file = "edge_weight.Rdata")
#####################











### Random Walk
# load("protein_POI.Rdata")
# protein

library('igraph')
load("gene_name.Rdata")
# protein_with_drug = c('TLR2', 'ITGB3', 'PTPN11', 'PIK3CA', 'CCL5', 'IL6', 'IL1B')
protein_with_drug = gene_name

# intersect(protein_with_drug, protein)


# DDI_node <- unique(as_ids(V(DDI)))
# 
# DTI_graph
# load('Node_diff.Rdata')
# unique(Node_diff)  # 16 drugs (no inter)
# 
# `%ni%` <- Negate(`%in%`)
# # DDI_seed <- DDI_node[DDI_node %ni% unique(Node_diff)]
# # Multiplex_Seeds <- DDI_seed
# 
# # Multiplex_Seeds <- unique(Node_diff)[1]
# # SecondNet_Seeds <- c(as_ids(V(PPI)[1]))
# 
# DDI_node <- DDI_node[order(DDI_node)]

load('drug_common.Rdata')
drug_common

RWR_record <- matrix(0, nrow = length(drug_common), ncol = length(drug_common) - length(protein_with_drug) + 3366)




# for (i in (1:length(drug_common))){
#   # print(i)
#   Multiplex_Seeds <- drug_common
#   # SecondNet_Seeds <- c(as_ids(V(PPI))[-6])
#   SecondNet_Seeds <- protein_with_drug
#   
#   RWR_MultiHetResults <- 
#     Random.Walk.Restart.MultiplexHet(MultiHetTranMatrix,multiplexHet,
#                                      Multiplex_Seeds ,SecondNet_Seeds,
#                                      MeanType = "Geometric")
#   RWR_hete <- RWR_MultiHetResults[["RWRMH_GlobalResults"]]
#   RWR_hete <- RWR_hete[order(RWR_hete$NodeNames),]
#   score <- RWR_hete$Score
#   node <- RWR_hete$NodeNames
#   RWR_record[i,] = score
# }




# intersect(drug_common, unique(drug_target_inter_data$parent_key) )
Multiplex_Seeds <- drug_common
SecondNet_Seeds <- protein_with_drug

RWR_MultiHetResults <- 
  Random.Walk.Restart.MultiplexHet(MultiHetTranMatrix,multiplexHet,
                                   ndoes ,target_node_DTI,
                                   MeanType = "Geometric")


RWR_hete <- RWR_MultiHetResults[["RWRMH_GlobalResults"]]
RWR_hete <- RWR_hete[order(RWR_hete$NodeNames),]
score <- RWR_hete$Score
node <- RWR_hete$NodeNames

# 
# node %>% select(starts_with("DB"))

load("jaccard_sim_label_0.Rdata")
jaccard_sim_468_DDI  ## Training
drug_RWR <- filter(RWR_hete, NodeNames %in% rownames(jaccard_sim_468_DDI))
prot_RWR <- filter(RWR_hete, NodeNames %in% node_name)




#####
#####
##### Random walk result on prot and drugs
drug_prot_RWR <- bind_rows(drug_RWR, prot_RWR) # drug + prot

# save(drug_prot_RWR, file = "drug_prot_RWR.Rdata")











RWR_record <- score
RWR_record <- as.data.frame(RWR_record)
rownames(RWR_record) <- node
rownames(RWR_record) <- make.names(node, unique=TRUE)

which(RWR_record!=0)

load("jaccard_sim_label_0.Rdata")
jaccard_sim_468_DDI  ## Training
rownames(jaccard_sim_468_DDI)
# save(jaccard_sim_other, file = "jaccard_sim_all_other_to_all.Rdata")  ### Other drugs
load("jaccard_sim_all_other_to_all.Rdata")
jaccard_sim_other  ## Testing
rownames(jaccard_sim_other)




# 
# # Traning
# # RWR_record_training <- filter(RWR_record, rownames(RWR_record) %in% rownames(jaccard_sim_468_DDI))
# RWR_record_training <- RWR_record[rownames(RWR_record) %in% rownames(jaccard_sim_468_DDI),  ]
# RWR_record_training <- as.data.frame(RWR_record_training)
# rownames(RWR_record_training) <- rownames(jaccard_sim_468_DDI)
# RWR_record_training
# 
# # Testing
# # RWR_record_testing <- 
# # RWR_record_testing <- filter(RWR_record, rownames(RWR_record) %in% rownames(jaccard_sim_other))
# RWR_record_testing <- RWR_record[rownames(RWR_record) %in% rownames(jaccard_sim_other),]
# RWR_record_testing <- as.data.frame(RWR_record_testing)
# rownames(RWR_record_testing) <- rownames(jaccard_sim_other)
# RWR_record_testing
# 
# # save(RWR_record, file = 'RWR_record_30_30.Rdata')
# # save(RWR_record, file = 'RWR_record_258_267.Rdata')
# # save(RWR_record, file = 'RWR_record_all.Rdata')
# 
# 
# # save(RWR_record_training, file = 'RWR_record_training.Rdata')
# # save(RWR_record_testing, file = 'RWR_record_testing.Rdata')

load('RWR_record_all.Rdata')
RWR_record
  


# 
# 
# DDI_plot <- delete.vertices(simplify(DDI), degree(DDI)<=470)
# Multiplex_DDI_plot <- create.multiplex(list(DDI = DDI_plot))  
# 
# 
# PPI_plot <- delete.vertices(simplify(PPI_Network_1), degree(PPI_Network_1)<=65)
# Multiplex_PPI_plot <- create.multiplex(list(DDI = PPI_plot)) 
# 
# 
# multiplexHet_visual <- create.multiplexHet(Multiplex_DDI_plot, Multiplex_PPI_plot, DTI_data_frame)


# degree(TopResults_PPI_Disease)

### Visualize Hete network # delete part for visulization
TopResults_PPI_Disease <-
  create.multiplexHetNetwork.topResults(RWR_MultiHetResults,
                                        multiplexHet, DTI_data_frame, k=1)

# V(DDI)$name[0:100]
# TopResults_PPI_Disease_visual <- delete.vertices(simplify(TopResults_PPI_Disease), degree(TopResults_PPI_Disease)<=40) %>% delete_vertices(drug_DDI_all[0:100])
# TopResults_PPI_Disease_visual <- delete.vertices(simplify(TopResults_PPI_Disease), V(DDI)[degree(DDI)<=100])
TopResults_PPI_Disease_visual <- delete.vertices(TopResults_PPI_Disease, V(TopResults_PPI_Disease) %in% V(DDI)[degree(DDI)<=450])
# V(TopResults_PPI_Disease) %in% V(DDI)[0:200]
TopResults_PPI_Disease_visual <- delete.vertices(TopResults_PPI_Disease_visual, degree(TopResults_PPI_Disease_visual)<=1)

length(E(TopResults_PPI_Disease_visual))
V(TopResults_PPI_Disease_visual)

E(TopResults_PPI_Disease_visual)$type

# lay <- layout_with_gem
lay <- layout_in_circle
Plott = TRUE


if(Plott){
  # par(mar=c(0.1,0.1,0.1,0.1))
  plot(TopResults_PPI_Disease_visual, vertex.label.color="black",
       vertex.frame.color="#ffffff",
       vertex.label.dist=0,
       vertex.label.font=1,
       vertex.label.cex=18,
       vertex.size= 20, 
       edge.curved=.2,
       vertex.color = ifelse(is.element(V(TopResults_PPI_Disease_visual)$name, SecondNet_Seeds), "orange",
                             ifelse(is.element(V(TopResults_PPI_Disease_visual)$name, Multiplex_Seeds) ,"lightblue","lightblue"
                                  # ifelse(V(TopResults_PPI_Disease)$name %in%
                                  #     multiplexHet$Multiplex1$Pool_of_Nodes,"#00CCFF","Grey75")
                                  )),

       edge.color=ifelse( E(TopResults_PPI_Disease_visual)$type == "PPI","orange",
                         ifelse( E(TopResults_PPI_Disease_visual)$type == "DDI","lightblue", "black"
                                )),
       edge.width=3,
       edge.lty=ifelse(E(TopResults_PPI_Disease_visual)$type == "bipartiteRelations",
                       5,1),
       vertex.shape= ifelse(V(TopResults_PPI_Disease_visual)$name %in%
                              multiplexHet$Multiplex1$Pool_of_Nodes,"circle","circle"),
       layout = lay  )
  legend(
    0.883,1.15,
    legend = c( "DDI", "PPI", "DTI"),
    col = c( "lightblue", "orange", "black"),
    lty = c(1, 1, 5),
    lwd = 3, 
    cex    = 20,
    bty    = "n"
  )
  
  legend(0.883,0.8, legend=c("Drug", "Protein"),
         col=c("lightblue", "orange"),cex=20,lwd=10,lty=c(0,0), pch = c(20, 20),bty="n")
}







