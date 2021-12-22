# BiocManager::install("pathview")

library('Biostrings')
library('biomartr') 
library("topGO")

setwd('c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation')  # \\ abs path
convt_gene <- read.delim("convt_gene.txt")
convt_list <- convt_gene$To
convt_list <- matrix(convt_list, ncol =, nrow =)

glist <- read.delim("gene_id.txt")


load('find_DEG_up.Rdata')
load('find_DEG_down.Rdata')
DEG_up = rownames(find_DEG_up)       # 
DEG_down = rownames(find_DEG_down)   # 
DEG_total = c(DEG_up,DEG_down)     # total 1967


library('org.Hs.eg.db')
convt_list <- mapIds(org.Hs.eg.db, DEG_total, 'ENTREZID', 'SYMBOL')
save(convt_list, file = "convt_list.RData")


load(file = "convt_list.RData")



###
###
### get GO term 
gene = as.character(convt_list)
ggo <- enrichGO(gene     = gene,
               OrgDb    = org.Hs.eg.db,
               keyType = 'ENTREZID',
               ont      = "BP",
               pvalueCutoff=0.01,
               qvalueCutoff  = 0.05,
               pAdjustMethod= "BH",
               readable = TRUE)

options(ggrepel.max.overlaps = Inf)

dotplot(ggo, split="ONTOLOGY") + facet_grid(ONTOLOGY~., scale="free")


par(cex = 0.18)
# plotGOgraph(ggo, firstSigNodes = 5, useInfo = c("def"), sigForAll = FALSE)
# showSigOfNodes(ggo,)
# 
# showSigOfNodes(ggo, similarity_semantic, firstSigNodes = 5, useInfo = 'all')
# printGraph(GOdata, resultFisher, firstSigNodes = 5, fn.prefix = "sampleFile", useInfo = "all", pdfSW = TRUE)
# 
# library("DOSE")
library(clusterProfiler)
# enrichMap(ggo,vertex.label.cex=1.2, layout=igraph::layout.kamada.kawai)







### Dotted line and solid line
par(cex = 0.1)
goplot(ggo,showCategory = 10, vertex.size=1.5)
par(cex = 0.1)





# ego <- simplify(ggo)
### https://yulab-smu.top/biomedical-knowledge-mining-book/enrichplot.html
# cnetplot(ggo, foldChange=convt_list, circular = TRUE, colorEdge = TRUE)
cnetplot(ggo, showCategory=10, node_label="all",foldChange=convt_list)

cnetplot(ggo, showCategory=10, node_label="category",foldChange=convt_list)
# 'cnetplot' depicts the linkages of genes and biological concepts (e.g. GO terms) as a network.
cnetplot(ggo, foldChange=convt_list, circular = TRUE, colorEdge = TRUE)
# plotGOgraph(ggo)




###
###
### GO Semantic similarity matrix

library(GOSemSim)
library('org.Hs.eg.db')
# hsGO <- godata('org.Hs.eg.db', ont="MF")
hsGO2 <- godata('org.Hs.eg.db', keytype = "SYMBOL", ont="BP", computeIC=FALSE)  # "BP" biological process, MF:Molecular Function, CC: Cellular Component 
genes <- DEG_total
# genes <- convt_list
genes <- c(genes,"PTPN11")

similarity_semantic <- mgeneSim(genes, hsGO2, measure="Wang")

save(similarity_semantic, file = 'similarity_semantic.Rdata')

nrow(similarity_semantic)

# similarity_semantic_1 <- mgeneSim(genes, hsGO2, measure="Wang", combine = "BMA")

load(file = 'similarity_semantic.Rdata')

SeedGene <- c("TLR2","ITGB3","CFP","PIK3CA","BAX","CCL5","IL6","IL1B","PTPN11") #"HLA-DMB","PILRB","ZFP37",

mylist <- c()
for( i in SeedGene){
  index = which(rownames(similarity_semantic) == i)
  mylist <- c(mylist, index)
}
# which(rownames(similarity_semantic) == SeedGene) 

row_sim <- similarity_semantic[mylist,]
col_sim <- row_sim[,mylist]


library(readxl)
drug_data <- read_excel("Drug_data.xlsx")
# drug_id <- na.omit(drug_data$Drug_bank)
target_name <- (drug_data$Target)[1:(61-5)]   #  5  12  1  7  2  14  15
drug_name <- drug_data$`Drug name`[1:(61-5)]  #  1:5, 6:17, 18, 19:25, 26:27, 28:41, 42:56
# smiles[is.na(smiles)] <- 0   # 66
# smiles_drug = smiles[1:61]   # 61
data <- matrix(nrow = length(drug_name), ncol = length(drug_name))
rownames(data) <- target_name
colnames(data) <- target_name
# table(target_name)
data[1:5, 1:5] = 1
data[6:17, 6:17] = 1
data[18:18, 18:18] = 1
data[19:25, 19:25] = 1
data[26:27, 26:27] = 1
data[28:41, 28:41] = 1
data[42:56, 42:56] = 1

# data[rownames(data)[1]]




###### GO similarity
for (i in rownames(data)){
  for (j in colnames(data)){
    
    row_index_sim = which(rownames(col_sim) == i)
    col_index_sim = which(colnames(col_sim) == j)
  

    row_index_data = which(rownames(data) == i)
    col_index_data = which(colnames(data) == j)
    if (  length(col_index_sim!=0 & row_index_sim!=0 &   col_index_sim==0  )!=0  ){
      data[row_index_data, col_index_data] = col_sim[row_index_sim, col_index_sim]
      
    }
  }
}


save(data, file = "drug_GO_similarity.Rdata")











data <- data.frame(1:length(drug_name),1:length(drug_name))
data_mat <- data.matrix(data)


for (name in target_name){
  rownames(col_sim)
}



### Pathway KEGG
###

library(clusterProfiler)

gene = as.character(convt_list)   ### 1967 genes mapping
kk <- enrichKEGG(
                  gene,
                  organism = "hsa",
                  keyType = "kegg",
                  # pvalueCutoff = 0.05,
                  # pAdjustMethod = "BH",
                  # universe,
                  # minGSSize = 10,
                  # maxGSSize = 500,
                  # qvalueCutoff = 0.05,
                  use_internal_data = FALSE,
                )
head(kk)  ### pathway "hsa05168"   Generatio: 79/768 495/8113
kk[,]

browseKEGG(kk, 'hsa05168')

library("pathview")

load("DEG_P.Rdata")

gene_data = as.double(DEG_P[,2])
gene_data = data.frame(gene_data)
gene_id_name = as.character(as.numeric(convt_list))
# row.names(gene_data) = make.names(c(convt_list), unique = TRUE)
# rownames(gene_data) = make.names(gene_id_name, unique = TRUE)
# rownames(gene_data) = as.numeric(convt_list)
test = as.double(unlist(gene_data))
# names(geneList)

names(test) <- c(convt_list)


hsa05168 <- pathview(gene.data  = test,
                     pathway.id = "hsa05168",
                     species    = "hsa",
                     split.group = TRUE,
                     limit=c(min(test), max(test)),
                     # kegg.native = FALSE,
                     same.layer = F           ### Nodes are original gene symbols
                     )

dim(hsa05168$plot.data.gene)










### Drug similarity
###
# # BiocManager::install("ChemmineR")
# # install.packages("dbparser")
# library(dbparser)
# library(dplyr)
# library(ggplot2)
# library(XML)
# 
# # devtools::install_github("interstellar-Consultation-Services/dbdataset")
# 
# # library(dbdataset)
# 
# getSmiFromDrugBank(id, parallel = 5)
# 
# 
# # drug = dbdataset
# 
# # BiocManager::install("BioMedR")
# install.packages("BioMedR")
# library(devtools)
# install_github('wind22zhu/BioMedR')
# # BiocManager::install("Rcpi", dependencies = c("Imports", "Enhances"))

library(Rcpp)
library(readxl)
drug_data <- read_excel("Drug_data.xlsx")
drug_id <- na.omit(drug_data$Drug_bank)
smiles <- drug_data$SMILES
smiles[is.na(smiles)] <- 0   # 66

smiles_drug = smiles[1:56]   # 56
# smile_matrix <- data.frame(smiles_drug)


# install.packages('stringdist')
library(stringdist)
jaccard_sim <- stringsimmatrix(smiles_drug,smiles_drug, method = "jaccard")
save(jaccard_sim, file = "jaccard_sim.Rdata")

















# select genes to shown their regulation with others
node.genes = c("ZNF641", "BCL6")
# enlarge the centrality
centrality.score = degree$centrality*100
names(centrality.score) = degree$Gene
par(mar = c(2,2,3,2))
grnPlot(grn.data = human.grn[[tissue]], cate.gene = cate.gene, filter = TRUE,
        nodes = node.genes, centrality.score = centrality.score,
        main = "Gene regulatory network")





####### Drug

library(dbparser)
library(XML)
## parse data from XML and save it to memory
read_drugbank_xml_db("drug_data_xml.xml")


# read_drugbank_xml_db("..path-to-DrugBank/full database.xml")

# load drugs data
drugs <- parse_drug() %>% select(primary_key, name)
drugs <- rename(drugs,drug_name = name)

## load drug target data
drug_targets <- parse_drug_targets() %>%
  select(id, name,organism,parent_key) %>%
  rename(target_name = name)

## load polypeptide data
drug_peptides <- parse_drug_targets_polypeptides()  %>%
  select(id, name, general_function, specific_function,
         gene_name, parent_id) %>%
  rename(target_name = name, gene_id = id)

# join the 3 datasets
drug_targets_full <- inner_join(drug_targets, drug_peptides,
                                by=c("id"="parent_id", "target_name")) %>%
  inner_join(drugs, by=c("parent_key"="primary_key")) %>%
  select(-other_keys)