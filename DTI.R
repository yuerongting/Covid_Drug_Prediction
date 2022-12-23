### DTI (search by target proteins from DrugBank data)
###
# DTI = sif2igraph("stitch_interactions.sif", directed = FALSE)
# plot(DTI, vertex.size=6)  
library(dplyr)
library(ggplot2)
library(OmnipathR)
library(igraph)
library(ggraph)
library(dbparser)
library(XML)
getwd()
setwd('c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation')


interactions = import_omnipath_interactions() %>% as_tibble()
# Convert to igraph objects:
OPI_g = interaction_graph(interactions = interactions )
get_xml_db_rows("drugbank.xml")


read_drugbank_xml_db("drugbank.xml")
# load All drugs data ---------------------------------------------------------------

# drugs <- dbparser::parse_drug() 
# drugs <- drugs() 
# save(drugs, file='drugs.Rdata')

load('drugs.Rdata')
drugs <- drugs$general_information
drugs_data <- drugs %>% as_tibble() %>% dplyr::select(primary_key, name, type, state, average_mass, monoisotopic_mass)
colnames(drugs_data)[2] <- "drug_name"

drug_targets_1 <- parse_drug_targets()
drug_targets_1 <- drug_targets_1[,c(1,2,3,6)]
## load polypeptide data

drug_peptides_1 <- parse_drug_targets_polypeptides()
drug_peptides_1 <- drug_peptides_1[,c(1,3,6,20)] # id, name, general_function, specific_function, gene_name, parent_id

# join the 3 datasets
drug_targets_full <- inner_join(drug_targets_1, drug_peptides_1,
                                by=c("id"="parent_id", "name")) # %>% inner_join(drugs, by=c("parent_key"="primary_key"))

drug_target_inter <- drug_targets_full %>% dplyr::select(parent_key, gene_name)


# save(drug_target_inter, file = "drug_target_inter.Rdata")  ### save All "drug_target_inter" from database




# Filter drugs ---------------------------------------------------------------

### protein of interest (POI)
load("drug_target_inter.Rdata")

drug_target_inter   ### All drugs

# load("chemical_drugs_name.Rdata")  ### chemical drugs  (Not match with SMILES?)
load("drug_calc_prop.Rdata")
# load("protein_POI.Rdata")
load("gene_name.Rdata")
# write.table(gene_name, file = "gene_name.txt", sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)

# protein
gene_name   ### Given 495 POI
# intersect(gene_name, protein)

protein = gene_name

drug_target_inter_data <- drug_target_inter   ### All DTI
drug_SMILES <- (drug_calc_prop[which(drug_calc_prop[,1] == "SMILES"  ),]) [, c(4,2)]  # All Drugs SMILES



drug_nam <- filter(drug_target_inter_data, gene_name %in% protein)  ### filter by given POI
drug_nam <- unique(drug_nam$parent_key)    # 291 drugs in DTI



# drug_common <- intersect(intersect(drug_nam, chemical_drugs_name), drug_SMILES$parent_key) ### Drugs (chemical, SMILE, DTI)
drug_common <- intersect(drug_nam, drug_SMILES$parent_key) ### Drugs (chemical, SMILE, DTI)

length(drug_common) # 214 drugs in common (manual & database)



### 295 DTI interactions

setdiff(drug_common, unique(as_ids(V(DDI)) )   )

drug_target_inter_data <- filter(drug_target_inter, parent_key %in% drug_common)
drug_target_inter_data <- filter(drug_target_inter_data, gene_name %in% protein)  ### 295 DTI

length(unique(drug_target_inter_data$parent_key))   ### 214 drugs
length(unique(drug_target_inter_data$gene_name))   ### 64 POI


load("drug_name.Rdata")
drug_name   ## 468 drugs

setdiff(unique(drug_target_inter_data$parent_key), DTI_data_frame$V1)


# save(drug_common, file = 'drug_common.Rdata')
# save(drug_target_inter_data, file = 'drug_target_inter_data.Rdata')

# save(drug_common, file = 'drug_common.Rdata')
# save(drug_target_inter_data, file = 'drug_target_inter_data_03_22.Rdata')



# DTI ---------------------------------------------------------------

# load("drug_target_inter_data.Rdata")
load("drug_target_inter_data_03_22.Rdata")
drug_target_inter_data      ### 295 DTI interactions

DTI_graph = graph_from_data_frame(drug_target_inter_data, directed = FALSE)   ### 278 nodes, 295 DTI edges
# save(drug_target_inter_data, file = 'DTI_graph.Rdata')


# load("drug_name.Rdata")
# drug_name

# load("drug_common.Rdata")
drug_nam <- unique(drug_target_inter_data$parent_key)    ### 214 drugs

prot_DTI <- unique(drug_target_inter_data$gene_name)    ### 64 drugs

# unique(drug_target_inter_data$parent_key)  ## 58 drugs
# intersect(unique(drug_target_inter_data$parent_key), drug_common)
# unique(intersect(rownames(jaccard_sim), drug_common))     
# unique(intersect(rownames(jaccard_sim), drug_SMILES$parent_key))
# unique(intersect(drug_common, drug_SMILES$parent_key))


library(RandomWalkRestartMH)
MultiplexObject <- create.multiplex(list(DTI = DTI_graph))  ### DTI: 278 nodes, 295 edges

AdjMatrix_DTI <- compute.adjacency.matrix(MultiplexObject)
DTI <- as.matrix(AdjMatrix_DTI)
DTI <- DTI[order(rownames(DTI)), order(colnames(DTI))]  ###  matrix 278 x 278
DTI <- DTI[startsWith(rownames(DTI),"DB"), !startsWith(colnames(DTI),"DB")] ### 214 x 64, row = drugs, col = proteins
rownames(DTI)<-gsub("_1","",rownames(DTI))
colnames(DTI)<-gsub("_1","",colnames(DTI))
# save(DTI, file = "DTI_known.Rdata")  ### Known DTI 214 x 64



add_zero_col <- matrix(0, nrow = nrow(DTI), ncol = abs( dim(DTI)[2] - length(protein)  ) )
add_zero_col <- as.data.frame(add_zero_col)

colnames(DTI)<-gsub("_1","",colnames(DTI))
rownames(DTI)<-gsub("_1","",rownames(DTI))

colnames(add_zero_col) <- setdiff(protein, colnames(DTI))

DTI <- cbind(DTI, add_zero_col)  ### DTI: 214 drug x 495 protein
rownames(DTI)<-gsub("_1","",rownames(DTI))

# save(DTI, file = "DTI_all.Rdata")



DTI_drug_order <- rownames(DTI)   # 214 drug names
# save(DTI_drug_order,file = "DTI_drug_order.Rdata")











# load("drug_SMILES_296.Rdata")
# load("drug_SMILES.Rdata")
# drug_SMILES_all
# 
# # DTI_drugs_30 <- filter(drug_SMILES, parent_key %in% rownames(DTI))
# 
# drug_SMILES$parent_key   ### 296 drugs
# drug_common   ### 214 drugs
# drug_nam <- unique(drug_target_inter_data$parent_key) ###  == drug_common
# 






###
###
### Save DTI_tibble (Python)
load("DTI_known.Rdata")  ### Known DTI 214 x 64
DTI
dim(DTI)

DTI_all.Rdata

load("DTI_all.Rdata")  ### Known DTI 214 x 495
DTI
dim(DTI)


load('drug_SMILES_468.Rdata')
drug_SMILES
drug_SMILES_468 = drug_SMILES
dim(drug_SMILES_468)




# load("jaccard_sim_DTI_drug.Rdata")
# dim(jaccard_sim_DTI_drugs)
# jaccard_sim
# 
# intersect(drug_nam, drug_SMILES$parent_key) # 214 drugs
# intersect(drug_nam, drug_common) # 214 drugs


### 296 - 30 
# add_zero_row <- matrix(0, nrow = length(drug_SMILES$parent_key) - dim(DTI)[1] , ncol = ncol(DTI))  ### 258 - 36 = 222 rows

add_zero_row <- matrix(0, nrow = length(drug_SMILES$parent_key) - dim(DTI)[1] , ncol = ncol(DTI) )  ### 468 - 214 = 254 rows
add_zero_row <- as.data.frame(add_zero_row)   ### 254 x 495



rownames(add_zero_row) <- setdiff(drug_SMILES$parent_key, rownames(DTI))  ### 254 x 495
colnames(add_zero_row) <- colnames(DTI)

# DTI <- rbind(DTI, add_zero_row)   # (228 + 36) x 9
# DTI <- DTI[order(rownames(DTI)), order(colnames(DTI))]  ###  matrix 37 x 37
DTI <- rbind(DTI, add_zero_row)   # 468 x 495
DTI <- DTI[order(rownames(DTI)), order(colnames(DTI))]  ###  468 x 495
# DTI <- subset(DTI, rownames(DTI) %in% rownames(jaccard_sim)  )
# DTI <- DTI[order(rownames(DTI)) , order(colnames(DTI))]

dim(DTI)

plot(DTI, vertex.size=1)

DTI_tibble <- as_tibble(DTI)
rownames(DTI_tibble) <- rownames(DTI)

# save(DTI, file = "DTI.Rdata")
# save(DTI_tibble, file = "DTI_tibble.Rdata")

