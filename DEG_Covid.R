getwd()
setwd('c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation')  # \\ abs path
# setwd("C:/Users/yrt05/Desktop/Systems Biology_Stroke-20210522T190533Z-001/Systems Biology_Stroke") # \\ abs path

#download data from GEO database, get file ".csv"
#function source: https://github.com/jmzeng1314/humanid/blob/master/R/downGSE.R
if(T){
  downGSE <- function(studyID = "GSE147507", destdir = ".") {
  library(GEOquery)
  eSet <- getGEO(studyID, destdir = destdir, getGPL = T)
  
  exprSet = exprs(eSet[[1]])
  pdata = pData(eSet[[1]])
  
  
  # save(eSet,file = 'GSE147507_eSet.Rdata')
  # write.csv(exprSet, paste0(studyID, "_exprSet.csv"))  
  # write.csv(pdata, paste0(studyID, "_metadata.csv"))  # classification of data
  return(eSet)
  }}


data <-read.table(file = 'GSE147507_RawReadCounts_Human.tsv', sep = '\t', header = TRUE)
# anota <-read.table(file = 'GSE147507_RawReadCounts_Human.tsv', sep = '\t', header = TRUE)

# BiocManager::install("GEOquery")
library(GEOquery)
eSet <- downGSE('GSE147507')
# save(eSet,file = 'eSet.Rdata')
# 
# ### load data
# load('GSE147507_eSet.Rdata')
# 
# ### get group names
# BiocManager::install("Biobase")
# library(Biobase)
data_group = pData(eSet[[1]])

pdata = data[,c(8:10,14:15,18:19,22:24,11:13,25:27)]      

# save(pdata,file = 'pdata.Rdata')

dim(pdata)
group_list=as.character(data_group[c(7:9,13:14,17:18,21:23,10:12,24:26),8])  

# group_list=colData(pdata)$dex
library(stringr)
group_list_replace <- gsub("\\s+", "_", str_trim(group_list))
group_list_replace <- gsub("-", "_", str_trim(group_list_replace))

### Normalization
# BiocManager::install("limma")
suppressMessages(library(limma))
design <- model.matrix(~0+factor(group_list_replace))
colnames(design)=levels(factor(group_list_replace))
rownames(design)=colnames(pdata)
# save(design,file='design.Rdata')

contrast.matrix<-makeContrasts("Mock_treated_A549_cells - SARS_CoV_2_infected_A549_cells",levels = design)

head(pdata)
rownames(pdata) = data[,1]
v <- voom(pdata,design,normalize="quantile")  # normalization

head(v[[1]])

fit <- lmFit(v,design)
fit2 <- contrasts.fit(fit, contrast.matrix)
fit3 <- eBayes(fit2)
tempOutput = topTable(fit3, coef=1, n=Inf)
DEG_voom = na.omit(tempOutput)
head(DEG_voom)

logFC_threshold = 0.58
# logFC_threshold = 1
p_val = 0.05
find_DEG_up = DEG_voom[which(DEG_voom[1]> logFC_threshold & DEG_voom[4]<p_val),]    # logFC = 0.58 is value 1.5
find_DEG_down = DEG_voom[which(DEG_voom[1]< (-logFC_threshold) & DEG_voom[4]<p_val),]

dim(find_DEG_up)
dim(find_DEG_down)
DEG_up = rownames(find_DEG_up)       # 913
DEG_down = rownames(find_DEG_down)  # 1054
DEG_total = c(DEG_up,DEG_down)     # total 1967
length(DEG_total) 

# # ### save DEG
# save(find_DEG_up,file = 'find_DEG_up.Rdata')
# save(find_DEG_down,file = 'find_DEG_down.Rdata')


# load('find_DEG_up_FC1.Rdata')
# load('find_DEG_down_FC1.Rdata')
# load('DEG_P.Rdata')
############################# find DEG 
# DEG_up = rownames(find_DEG_up)       # 274
# DEG_down = rownames(find_DEG_down)   # 444
# DEG_total = c(DEG_up,DEG_down)     # total 718
# length(DEG_total) 



# DEG_up_p = cbind( rownames(find_DEG_up) , find_DEG_up$logFC )
# DEG_down_p = cbind( rownames(find_DEG_down) , find_DEG_down$logFC )
# DEG_P = rbind(DEG_up_p, DEG_down_p)

# write(DEG_total,"DEG_total.txt",sep = '')  # save TXT data
# 
# write(DEG_P,"DEG_P.txt",sep = '')  # save TXT data
# save(DEG_P, file = "DEG_P.RData")









data_sample_probe <- v[[1]]
data_sample_probe <- as.data.frame(data_sample_probe)
library(dplyr)
DEG_voom %>% distinct()
DEG_voom <- DEG_voom[!duplicated(rownames(DEG_voom)), ] 

data_sample_probe <- data_sample_probe[!duplicated(rownames(data_sample_probe)), ] 

# BiocManager::install("org.Hs.eg.db")
library(org.Hs.eg.db)

# gene_probe_name <- mapIds(org.Hs.eg.db, keys = rownames(DEG_voom), keytype = "ENSEMBL", column = "SYMBOL", multiVals="first")
gene_probe_name <- rownames(DEG_voom)
gene_probe_name <- gene_probe_name[!duplicated(gene_probe_name)] 

inds <- which(!is.na(gene_probe_name))
found_genes <- gene_probe_name[inds]
df2 <- DEG_voom[found_genes, ]
rownames(df2) <- found_genes

data_sample_probe <- data_sample_probe[found_genes, ]   # For use in "Co-expression"
# rownames(data_sample_probe) <- found_genes
# save(data_sample_probe,file='data_sample_probe.Rdata')  # For use in "Co-expression"




### filter probes
# ids = toTable(mouse430a2SYMBOL) 
# table(rownames(df2) %in% ids$symbol)
exprSet <- data_sample_probe
# dat=exprSet
# exprSet=dat
### colorful boxplot
par(cex = 0.7)
n.sample=ncol(exprSet)
if(n.sample>40) par(cex = 0.5)
cols <- rainbow(n.sample*1.2)
boxplot(exprSet, col = cols,main="expression value",las=2)
# boxplot(exprSet[,5])





# BiocManager::install("KEGGREST")
# BiocManager::install("EnrichmentBrowser")
library("KEGGREST")
library("EnrichmentBrowser")

hsa_path_eg  <- keggLink("pathway", "hsa") %>% 
  tibble(pathway = ., eg = sub("hsa:", "", names(.)))
hsa_kegg_anno <- hsa_path_eg %>%
  mutate(
    symbol = mapIds(org.Hs.eg.db, eg, "SYMBOL", "ENTREZID"),
    ensembl = mapIds(org.Hs.eg.db, eg, "ENSEMBL", "ENTREZID")
  )
hsa_kegg_anno_hsa05168 <- hsa_kegg_anno[hsa_kegg_anno$pathway == 'path:hsa05168',]

length(intersect(hsa_kegg_anno$symbol, DEG_total)) ### 777 DEGs are mapped to database

DEG_hsa05168 <- intersect(hsa_kegg_anno_hsa05168$symbol, DEG_total)
# write(DEG_hsa05168, "DEG_hsa05168.txt")



############### heatmap
library(pheatmap)
choose_gene = head(DEG_total,30)
choose_matrix = exprSet[choose_gene,]
choose_matrix = t(scale(t(choose_matrix)))
pheatmap(choose_matrix, )
exprSet['Ptrh1',]















############### volcano plot
#### use DEG_voom not DEG
DEG=nrDEG

# DEG = DEG_P
DEG$logFC
# logFC_threshold = 0.58
# p_val = 0.05
# logFC_cutoff <- with(DEG,mean(abs( logFC)) + 2*sd(abs( logFC)) )
DEG$change = as.factor(ifelse(DEG$P.Value < p_val & abs(DEG$logFC) > logFC_threshold,
                              ifelse(DEG$logFC > logFC_threshold ,'Up-regulated','Down-regulated'),'NOT')
)

DEG_voom$Change = as.factor(ifelse(DEG_voom$P.Value < p_val & abs(DEG_voom$logFC) > logFC_threshold,
                              ifelse(DEG_voom$logFC > logFC_threshold ,'Up-regulated','Down-regulated'),'Not Significant')
)
# round(logFC_cutoff,3)
this_tile <- paste0('Cutoff:' , 'Fold Change > 1.5,', 'P Value < ', p_val,
                    '\nNumber of up-regulated gene: ',nrow(DEG[DEG$change =='Up-regulated',]) ,
                    '\nNumber of down-regulated gene: ',nrow(DEG[DEG$change =='Down-regulated',])
)

font_size = 15

this_tile
# install.packages("ggplot2")
library("ggplot2")
head(DEG)
DEG_voom
g = ggplot(data=DEG_voom, aes(x=logFC, y=-log10(P.Value), color=Change)) +
  geom_point(alpha=0.4, size=1.75) +
  theme_set(theme_set(theme_bw(base_size=font_size)))+
  xlab("log2 fold change") + ylab("-log10 P-value") +
  ggtitle( this_tile ) + theme(plot.title = element_text(size=font_size,hjust = 0))+
  scale_colour_manual(values = c('blue','black','red')) + 
  geom_hline(yintercept= -log10(p_val) -0.05, linetype="dashed", color = "black", size=2) + 
  geom_vline(xintercept = logFC_threshold - 0.05, linetype="dashed", color = "black", size=2) +
  geom_vline(xintercept = -logFC_threshold + 0.05, linetype="dashed", color = "black", size=2) + 
  theme(
    axis.title.x = element_text(size = font_size),
    axis.text.x = element_text(size = font_size),
    axis.title.y = element_text(size = font_size),
    axis.text.y = element_text(size = font_size),
    legend.title = element_text(size=font_size),
    legend.text = element_text(size=font_size)
    )
print(g)





########################## online pathway analyses

### STRING
# BiocManager::install("STRINGdb")
library(STRINGdb)

data(diff_exp_example1)
head(DEG_voom)
rownames(DEG_voom)
nrDEG <- DEG_voom%>%filter(rownames(DEG_voom)%in%DEG_total)
gene_List = c(nrDEG$P.Value,nrDEG$logFC)
names(gene_List) = rownames(nrDEG)


# string_db <- STRINGdb$new(version="11.5", species=9606, score_threshold=700, input_directory= '') # 10090 Mus musculus
# # deg_mapped <- string_db$map(together, "gene", removeUnmappedRows = TRUE )
# # string_db <- STRINGdb$new()
# head(as.data.frame(DEG_total))
# example1_mapped = string_db$map(as.data.frame(DEG_total), "DEG_total", removeUnmappedRows = TRUE )
# string_db$plot_ppi_enrichment( example1_mapped$STRING_id[1:1000] )
# 
# head(example1_mapped)



# ### mapping 
# if(F){
#   if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
#   BiocManager::install("GO.db")
#   } # install GO database




# library(R.utils);
# kegg_link <- function (target_db, source_db) 
# {
#   R.utils::setOption("clusterProfiler.download.method",'wget')
#   options(clusterProfiler.download.method = "wget")
#   getOption("clusterProfiler.download.method")
#   
#   url <- paste0("http://rest.kegg.jp/link/", target_db, "/", 
#                 source_db, collapse = "")
#   print(url)
#   clusterProfiler:::kegg_rest(url)
# }
# 
# R.utils::setOption("clusterProfiler.download.method",'wget')
# options(clusterProfiler.download.method = "wget")
# getOption("clusterProfiler.download.method")
# reassignInPackage("kegg_link", pkgName="clusterProfiler", kegg_link)
# 
# # BiocManager::install("clusterProfiler")
# R.utils::setOption("clusterProfiler.download.method",'auto')
# # BiocManager::install("clusterProfiler", version = "3.12")
# library(clusterProfiler)
# kk <- enrichKEGG(gene = gene_getname.df$ENTREZID,
#                  organism = "hsa",
#                  pvalueCutoff = 0.05)
# 
# 
# library(clusterProfiler)
# getOption("clusterProfiler.download.method")
# 
# 
# 
# 
# 
# 






####################### enrichment KEGG
library(clusterProfiler)
# data(geneList, package="DOSE") # Disease Ontology Semantic and Enrichment analysis
gene_getname.df <- bitr(DEG_total, fromType = "SYMBOL",
                        toType = c("ENTREZID"),
                        OrgDb = "org.Hs.eg.db")
head(gene_getname.df)


library(org.Hs.eg.db)
hs <- org.Hs.eg.db
ent_id <- select(hs, 
             keys = DEG_total,
             columns = c("ENTREZID", "SYMBOL"),
             keytype = "SYMBOL")
ent_id$ENTREZID
# data(geneList, package="DOSE")
# head(geneList)
# gene <- names(geneList)[abs(geneList) > 2]
# kk <- enrichKEGG(gene         = gene,
#                  organism     = 'hsa',
#                  pvalueCutoff = 0.05)


# install.packages("KEGG.db_3.2.4.tar.gz", repos = NULL, type = "source")
kk <- enrichKEGG(gene = ent_id$ENTREZID,
                 organism = "hsa",
                 pvalueCutoff = 0.05)



font_size = 15
dotplot(kk, showCategory=10, x = 'Count', font.size = font_size, orderBy="x")



GO_enrich <- enrichGO(gene_getname.df$ENTREZID, OrgDb = "org.Hs.eg.db", ont="all")












### Pathway analysis plot

head(gene_getname.df)


font_size = 11
library(scales) 
head(kk@result)
kk@result[["Description"]][1:3]
kegg <- kk
p<- dotplot(kegg, showCategory=5, x = 'Count', font.size = font_size, orderBy="x")

p1 <- p + theme(
  plot.title = element_text(color="black", size=font_size, face="bold", hjust=0.5))



gene_id <- bitr(rownames(nrDEG), fromType = "SYMBOL",
                toType = c("ENTREZID"),
                OrgDb = "org.Hs.eg.db")
fold_vector <- nrDEG$logFC
names(fold_vector) <- gene_id$ENTREZID


### Pathway
p2 <- cnetplot(kk, showCategory=5, foldChange=fold_vector, node_label = "category", layout = "kk",
               colorEdge = TRUE, color_category ='steelblue', node_label_size = NULL,
               cex_category = 1, label_format = 5, cex_label_category = 1)
# min.value <- floor( min(p2$data$color, na.rm = TRUE) )
# max.value <- ceiling( max(p2$data$color, na.rm = TRUE) )
# p2 <- p2 + scale_color_gradientn(name = "Fold Change",
#                            colours = c("blue", "red"),
#                            values = rescale(c(min.value, 0, max.value)),
#                            limits= c(min.value, max.value), 
#                            breaks=c(min.value , 0, max.value) )
p2 <- p2 + theme(legend.position="right",
                 legend.text = element_text(size = font_size),
                 # legend.background = element_rect(fill="lightblue", size = 1),
                 legend.title = element_text(size = font_size, face = "bold"),
                 legend.justification = c("right", "top"),
                 # legend.margin = margin(6, 6, 6, 6)
)
library('cowplot')
plot_grid(p2, p1, labels = c("a","b"), label_size = 15,  nrow = 2, align = c("h"))




### Significant DEGs
significant_gene_id <- kk@result[["geneID"]][1]

significant_gene_id <- gsub("/"," ", str_trim(significant_gene_id))

significant_gene_id <- str_split(significant_gene_id, " ")[[1]]

significant_gene <- bitr(significant_gene_id, fromType = "ENTREZID",
                         toType = c("SYMBOL"),
                         OrgDb = "org.Hs.eg.db")
significant_gene_KEGG <- significant_gene$SYMBOL


intersect(significant_gene_KEGG, selected_genes_name)

intersect(intersect(significant_gene_KEGG, selected_genes_name), DEG_WGCNA_common)


up_DEG <- intersect(significant_gene_KEGG, rownames(find_DEG_up))
down_DEG <- intersect(significant_gene_KEGG, rownames(find_DEG_down))
up_DEG <- bitr(up_DEG, fromType = "SYMBOL",
               toType = c("ENTREZID"),
               OrgDb = "org.Hs.eg.db")
down_DEG <- bitr(down_DEG, fromType = "SYMBOL",
                 toType = c("ENTREZID"),
                 OrgDb = "org.Hs.eg.db")
rownames(nrDEG)
P_up <- dat[rownames(dat) %in% up_DEG$SYMBOL,4]
P_down <- dat[rownames(dat) %in% down_DEG$SYMBOL,4]



### GO enrichment
# 
# Genes<-c(significant_gene_id)
# # g <- goana(Genes)
# # BiocManager::install("org.mmu.eg.db")
# # library(org.mmu.eg.db)
# g <- goana(list(Up=up_DEG$ENTREZID, Down=down_DEG$ENTREZID, P.Up = P_up, P.Down = P_down), 
#            species="Mm", covariate = TRUE, plot = TRUE)
# GO_result <- topGO(g)
# GO_result$Term
# 
# GO_result$

### GO enrichment 2
# library(clusterProfiler)
# # de <- c(up_DEG$ENTREZID, down_DEG$ENTREZID)
# de <- bitr(rownames(nrDEG), fromType = "SYMBOL",
#            toType = c("ENTREZID"),
#            OrgDb = "mouse430a2.db")
# de <- de$ENTREZID
# go <- enrichGO(de, OrgDb = "org.Mm.eg.db", ont="all")
# # ego <- enrichGO(de, OrgDb = "org.Mm.eg.db", ont="BP", readable=TRUE)
# # go <- enrichDAVID(de, idType="ENTREZ_GENE_ID",  listType="Gene", annotation="KEGG_PATHWAY")
# library(ggplot2)





library(GOplot)
data(EC)
test <- EC$david
# setwd("C:/Users/yrt05/Desktop/Systems biology project/GSE137482_RAW")
setwd('c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation')
readfile <- read.csv("gProfiler_mmusculus.csv")
readfile <- subset(readfile, select = c("source","term_name", "intersections", "adjusted_p_value", "term_id") )
colnames(readfile)[1:4] <-colnames(test)[c(1,3,4,5)]
readfile <- readfile[, c(1,5,2,3,4)]
colnames(readfile)[2] <-colnames(test)[2]
test3 <- as.factor(readfile$Genes)
length(test3)
test3 <- str_split(test3, ",")
# unlist(test3)
library("org.Mm.eg.db")
nrDEG_data <- cbind(rownames(nrDEG), nrDEG)
colnames(nrDEG_data) <- c("ID", colnames(nrDEG))
readfile$Category <- gsub("GO:","", str_trim(readfile$Category))
rownames(nrDEG_data) <- c(1:nrow(nrDEG_data))

# length(test3)
for (i in 1:length(test3)){
  map <- mapIds(org.Mm.eg.db, keys = unlist(str_split(unlist(test3[i]),",")) , keytype = "ENSEMBL", column = "SYMBOL", multiVals="first")
  # map <- mapIds(org.Mm.eg.db, keys = map , keytype = "SYMBOL", column = "PROBEID", multiVals="list")
  result <- data.frame(paste(toupper(map), collapse=', '))
  readfile$Genes[i] <- result
  # print(as.factor(paste(toupper(map), collapse=', ')))
}

readfile <- filter(readfile, Category %in% c("CC", "BP", "MF"))

circ <- circle_dat(readfile, nrDEG_data)


# GOBubble(circ, labels = 10)
# Reduce redundant terms with a gene overlap >= 0.75...
# reduced_circ <- reduce_overlap(circ, overlap = 0.75)
# # ...and plot it
# GOBubble(reduced_circ, labels = 2.8)

# GOBubble(circ, title = 'Bubble plot', colour = c('orange', 'darkred', 'gold'), display = 'multiple', labels = 10)  

# GOBar(subset(circ, category == 'BP'))







### GO plot

gene_getname.df

go <- GO_enrich

go2 <- dotplot(GO_enrich, showCategory=5, split="ONTOLOGY") + facet_grid(ONTOLOGY~., scale="free")

### Separate analyses
go_result <- summary(go)
go_result_All <- go_result$ID[order(go_result$Count, decreasing=T)]
# head(go_result_All)
go_result_All[1:5]
IDs <- c(go_result_All[1:5])
# install.packages('GOplot')
library(GOplot)
go1 <- GOCircle(circ, nsub = IDs,  rad1 = 2, rad2 = 3,  zsc.col = c('yellow', 'black', 'cyan'))

# go_result_BP <- filter(go_result, ONTOLOGY %in% c("BP"))
# go_result_BP <- go_result_BP$ID[order(go_result_BP$Count, decreasing=T)]
# go_result_MF <- filter(go_result, ONTOLOGY %in% c("MF"))
# go_result_MF <- go_result_MF$ID[order(go_result_MF$Count, decreasing=T)]
# go_result_CC <- filter(go_result, ONTOLOGY %in% c("CC"))
# go_result_CC <- go_result_CC$ID[order(go_result_CC$Count, decreasing=T)]


go2 <- go2 + theme(legend.position="right",
                   legend.text = element_text(size = font_size),
                   # legend.background = element_rect(fill="lightblue", size = 1),
                   legend.title = element_text(size = font_size, face = "bold"),
                   legend.justification = c("right", "top"),
                   # legend.margin = margin(6, 6, 6, 6)
)
# plot(go2)
library(cowplot)
# ,   rel_heights = c(0.3, 1.2)
# par(mfrow=c(1,2))
# dotplot(go, showCategory=5, split="ONTOLOGY") + facet_grid(ONTOLOGY~., scale="free")
# GOCircle(circ, nsub = IDs, label.size = 6, rad1 = 3, rad2 = 4,  zsc.col = c('yellow', 'black', 'cyan'))

plot_grid(go2, go1, labels = c("a","b"), label_size = 22, ncol = 2, align = c("h"), greedy = TRUE, 
          rel_widths = c(1, 1.5),
          axis = "b", scale = c(1,1), cex= 2)









