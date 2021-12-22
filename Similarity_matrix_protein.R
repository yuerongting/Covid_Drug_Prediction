# BiocManager::install("impute")

library("protr")
setwd('c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation')  # \\ abs path

# # load FASTA files
# extracell <- readFASTA("uniprot.fasta")
# 
# extracell_test <- readFASTA("uniprot-.fasta")


library("readxl")

## 659 out of 718 were mapped to protein based on UniProt database
protein_id <- read_excel("uniprot-protein_id.xlsx")

protein_id <- protein_id$Entry

prots <- getUniProt(protein_id)   # Protein Sequence

save(prots, file = "prots.RData")


names(prots) = protein_id

# prots_data = prots[1:10]
prots_data = prots



# psimmat <- parSeqSim(prots_data, cores = 6, type = "local", batches = 20, submat = "BLOSUM62", verbose = TRUE)

psimmat <- parSeqSimDisk(prots_data, cores = 6, type = "local", batches = 200, submat = "BLOSUM62", verbose = TRUE)
save(psimmat, file = "psimmat.RData")
# the amino acid type sanity check and remove the non-standard sequences, To ensure that the protein sequences only have the 20 standard amino acid types which is usually required for the descriptor computation
# remove 1 protein sequences

load("psimmat.RData")
load("prots.RData")

protein_seq_similarity <- psimmat
rownames(protein_seq_similarity) <- protein_id
colnames(protein_seq_similarity) <- protein_id

save(protein_seq_similarity, file = "protein_seq_similarity.RData")


load("protein_seq_similarity.RData")
protein_seq_similarity










### Drug data
Download_data = FALSE
if (Download_data){
  # Load dbparser package
  library(dbparser)
  # Create SQLite database connection
  database_connection <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
  # # DrugBank database sample name
  # biotech <- "drugbank.xml"
  # Use DrugBank database in the library
  read_drugbank_xml_db("drugbank.xml")
  
  # Parse all available drug tibbles
  run_all_parsers(save_table = TRUE, database_connection = database_connection)
  
  
  # List saved tables
  DBI::dbListTables(database_connection)
  # Close SQLite connection
  DBI::dbDisconnect(database_connection)
  ## load drugs data
  drugs <- drugs()
  
  ## load drug groups data
  # drug_groups <- drug_groups()
  
  drug_interaction <- drug_interactions()
  
  # drug_sequences()  ### only for biotech drugs
  
  drug_salts <- drug_salts()  ### similarity SMILES
  
  drug_calc_prop <- drug_calc_prop()
  
  
  save(drugs, file = "drugs.Rdata")
  save(drug_interaction, file = "drug_interaction.Rdata")
  save(drug_salts, file = "drug_salts.Rdata")
  save(drug_calc_prop, file = "drug_calc_prop.Rdata")
}


load("drugs.Rdata")
load("drug_interaction.Rdata")
load("drug_salts.Rdata")
load("drug_calc_prop.Rdata")








### DDI
###
library(Rcpp)
library(readxl)
drug_data <- read_excel("Drug_data.xlsx")
# drug_chemical <- na.omit(drug_data$SMILES)
# 
# myli <- c()
# for (i in 1:(length(drug_chemical)) ){
#   index <- which(drug_data$SMILES == drug_chemical[i])
#   myli <- c(myli, index)
# }
# dr <- drug_data$Drug_bank[myli]    #
# dr_name <- dr[!duplicated(dr)]
# 
# drug_id <- dr_name    # 32 drugs have SMILES

drug_id <- na.omit(drug_data$Drug_bank)[1:56]# 56 drugs of interest
drug_id <- drug_id[!duplicated(drug_id)] 


# smiles <- drug_data$SMILES
# smiles[is.na(smiles)] <- 0   # 66

# smiles_drug = smiles[1:61]   



drug_interact_1 <- drug_interaction[which(drug_interaction[,1] == drug_id), ] [, c(1,4)]   # DDI: 277 interactions
drug_interact_2 <- drug_interaction[which(drug_interaction[,4] == drug_id), ] [, c(1,4)]
drug_interact <- rbind(drug_interact_1, drug_interact_2)

# drug_name_no_repeat <- as.vector(rbind(drug_interact_1[,1], drug_interact_1[,2], drug_interact_2[,1], drug_interact_2[,2]))

library(dplyr)
drug_name_no_repeat <- (bind_rows(drug_interact_1[,1], drug_interact_1[,2], drug_interact_2[,1], drug_interact_2[,2] ))[,1]

drug_name <- drug_name_no_repeat[!duplicated(drug_name_no_repeat),] 

#       
# drug_name <- drug_interact[,1]
# drug_name <- drug_name[!duplicated(drug_name),]   # Only 24 drugs (chemical, not biotech)



# for (d in drug_interaction[,1] )  {
#   if(d == drug_id){
#     row_index_data = which(drug_id == drug_interaction[,1])
#   }
# }



### Drug SMILES similarity
###
drug_SMILES_name <- drug_name    ### name of drug similarity

# typeof(drug_id[i])
# typeof(toString(drug_SMILES_name[i,]))
# dim(drug_SMILES_name)
# 
# which(drug_SMILES[,1] ==drug_SMILES_name)

drug_SMILES <- (drug_calc_prop[which(drug_calc_prop[,1] == "SMILES"),]) [, c(4,2)]
mylist <- c()
for (i in 1:(dim(drug_SMILES_name)[1]) ){
  index <- which(drug_SMILES[,1] == toString(drug_SMILES_name[i,]))
  mylist <- c(mylist, index)
}
drug_SMILES <- drug_SMILES[mylist,]  ### With drug_id,   241 SMILES match

SMILES_data <- as.matrix(drug_SMILES[,2])  ## Only SMILES vector


library(stringdist)
jaccard_sim <- stringsimmatrix(SMILES_data,SMILES_data, method = "jaccard")  #  250 by 250
rownames(jaccard_sim) <- drug_SMILES$parent_key
colnames(jaccard_sim) <- drug_SMILES$parent_key

head(jaccard_sim)
save(jaccard_sim, file = "jaccard_sim.Rdata")









### DTI 
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

interactions = import_omnipath_interactions() %>% as_tibble()
# Convert to igraph objects:
OPI_g = interaction_graph(interactions = interactions )

get_xml_db_rows("drugbank.xml")

## load drugs data
drugs <- parse_drug() %>% select(primary_key, name)
drugs <- rename(drugs,drug_name = name)

## load drug target data
drug_targets <- parse_drug_targets() %>%
  select(id, name,organism,parent_key) %>%
  rename(target_name = name)

drug_targets_1 <- parse_drug_targets()

## load polypeptide data
drug_peptides <- parse_drug_targets_polypeptides()  %>%
  select(id, name, general_function, specific_function,
         gene_name, parent_id) %>%
  rename(target_name = name, gene_id = id)

drug_peptides_1 <- parse_drug_targets_polypeptides()

# join the 3 datasets
drug_targets_full <- inner_join(drug_targets, drug_peptides,
                                by=c("id"="parent_id", "target_name")) %>%
  inner_join(drugs, by=c("parent_key"="primary_key"))



drug_target_inter <- drug_targets_full %>% select(parent_key, gene_name)


### protein of interest (POI)
protein = c("TLR2","ITGB3","CFP","PIK3CA","BAX","CCL5","IL6","IL1B","PTPN11")
mylistt<- c()
for (i in 1:length(POI)){
  index <- which(drug_target_inter$gene_name == protein[i])
  mylistt <- c(mylistt, index)
}
drug_target_inter_data <- drug_target_inter[mylistt,]

save(drug_target_inter_data, file='drug_target_inter_data.Rdata')














# drugnames = drug_SMILES[,1]$parent_key          ### Drug of interest
drug_names = drug_id          ### Drug of interest
# drug_name$`drugbank-id`
mylis<- c()
for(i in 1:length(drug_id)){
  index <- which(drugs[,1]==drug_id[i])
  mylis <- c(mylis,index)
}
drug_names = drugs[mylis,]


# drug_name<-c()
drug_target_data_sample <- drug_targets_full %>%
  filter(organism == "Humans",drug_name %in% drug_names$drug_name)

# drugnames <-drug_names$drug_name
# drug_targets <- OmnipathR:::drug_target_data_sample %>%
#   filter(organism == "Humans",drug_name %in% drug_names)




drug_targets <-  drug_target_data_sample %>%
  select(-specific_function, -organism, -general_function) %>%
  mutate(in_OP = gene_id %in% c(interactions$source))
# not all drug-targets are in OP.
print(all(drug_targets$in_OP))

drug_targets %>% group_by(parent_key) %>% summarise(any(in_OP))




### protein of interest (POI)
POI = tibble(protein = c("TLR2","ITGB3","CFP","PIK3CA","BAX","CCL5","IL6","IL1B","PTPN11") )

POI <- POI %>% mutate(in_OP = protein %in% interactions$target_genesymbol)
# all POI is in Omnipath
print(all(POI$in_OP))







### 
drug_of_interest = "Abciximab"
source_nodes <- drug_targets %>%
  filter(in_OP, drug_name==drug_of_interest) %>%
  pull(gene_name)
target_nodes <- POI %>% filter(in_OP) %>% pull(protein)

collected_path_nodes = list()

for(i_source in 1:length(source_nodes)){
  
  paths <- shortest_paths(OPI_g, from = source_nodes[i_source],
                          to = target_nodes,
                          output = 'vpath')
  path_nodes <- lapply(paths$vpath,names) %>% unlist() %>% unique()
  collected_path_nodes[[i_source]] <- path_nodes
}
collected_path_nodes <- unlist(collected_path_nodes) %>% unique()



cisplatin_nodes <- c(source_nodes,target_nodes, collected_path_nodes) %>%
  unique()
cisplatin_network <- induced_subgraph(graph = OPI_g,vids = cisplatin_nodes)


V(cisplatin_network)$node_type = ifelse(
  V(cisplatin_network)$name %in% source_nodes, "direct drug target",
  ifelse(
    V(cisplatin_network)$name %in% target_nodes,"Protein of Interest","intermediate node"))

ggraph(
  cisplatin_network,
  layout = "lgl",
  area = vcount(cisplatin_network)^2.3,
  repulserad = vcount(cisplatin_network)^1.2,
  coolexp = 1.1
) +
  geom_edge_link(
    aes(
      start_cap = label_rect(node1.name),
      end_cap = label_rect(node2.name)),
    arrow = arrow(length = unit(4, 'mm')
    ),
    edge_width = .5,
    edge_alpha = .2
  ) +
  geom_node_point() +
  geom_node_label(aes(label = name, color = node_type)) +
  scale_color_discrete(
    guide = guide_legend(title = 'Node type')
  ) +
  theme_bw() +
  xlab("") +
  ylab("") +
  ggtitle("Abciximab induced network")

