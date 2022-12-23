rm(list = ls())
getwd()

### load data
load('pdata.Rdata')
data <-read.table(file = 'GSE147507_RawReadCounts_Human.tsv', sep = '\t', header = TRUE)
pdata = data[,c(8:10,14:15,18:19,22:24,11:13,25:27)]   
rownames(pdata) <- data[,1]
exprSet = pdata

dat = exprSet
# pdata = pData(eSet[[1]])[c(1:9,19:21,25:27,22:24,28:30),]

# levels(exprSet)
# a = exprSet[, -c(1:8)]
datExpr0 <- as.data.frame(t(exprSet))
# datExpr1 <- as.data.frame((exprSet))

datExpr0_trans = t(datExpr0)
m.vars=apply(datExpr0_trans,1,var)
datExpr0_trans.upper=datExpr0_trans[which(m.vars>quantile(m.vars, probs = seq(0, 1, 0.20))[5]),]##选择方差最大的前20%个基因作为后续WGCNA的输入数据集

datExpr0 = as.data.frame(t(datExpr0_trans.upper))    # left 4360 genes with large var
dim(datExpr0)

library('WGCNA')
gsg <- goodSamplesGenes(datExpr0, verbose=3)


### filter samples and probes: Good samples, some bad probes
if (!gsg$allOK){
  if (sum(!gsg$goodGenes)>0)
    printFlush(paste("Removing genes:", paste(names(datExpr0)[!gsg$goodGenes], collapse=", ")))
  if (sum(!gsg$goodSamples)>0) 
    printFlush(paste("Removing samples:", paste(rownames(datExpr0)[!gsg$goodSamples], collapse=", ")))
  datExpr0 <- datExpr0[gsg$goodSamples, gsg$goodGenes]
}

# sampleTree <- hclust(dist(datExpr0), method="average")
# sizeGrWindow(12, 9)
# par(cex=0.6)
# par(mar=c(0,4,2,0))
# plot(sampleTree, main="Sample clustering to detect outliers", sub="", xlab="", cex.lab=1.5, 
#      cex.axis=1.5, cex.main=2)
# abline(h=15, col="red")





# sample information table
load('GSE147507_eSet.Rdata')
load('eSet.Rdata')
### get group names
library(Biobase)
data_group = pData(eSet[[1]])

   


dim(pdata)
group_list=as.character(data_group[c(7:9,13:14,17:18,21:23,10:12,24:26),8])  

load('design.Rdata')
coldata <- design
colnames(coldata)[1] <- "Condition"
colnames(coldata)[2] <- "None"
coldata[,1] = group_list
cts = t(datExpr0)
all(rownames(coldata) %in% colnames(cts) ) 

dim(cts)                 # 4360 genes
all(rownames(coldata) == colnames(cts))



library("DESeq2")
### Normalization
dds <- DESeqDataSetFromMatrix(countData = cts,
                              colData = coldata,
                              design = ~ Condition)

dds_norm <- vst(dds)

normalized_counts <- assay(dds_norm)

normalized_counts_t <- t(normalized_counts)


sampleTree = hclust(dist(normalized_counts_t), method = "average")
plot(sampleTree, main = "Sample clustering to detect outliers", sub="", xlab="")

library(WGCNA)
### no obvious outliers
sampleTree <- hclust(dist(normalized_counts_t), method="average")
sizeGrWindow(12, 9)
par(cex=0.6)
par(mar=c(0,4,2,0))
plot(sampleTree, main="Sample clustering to detect outliers", sub="", xlab="", cex.lab=1.5, 
     cex.axis=1.5, cex.main=2)
abline(h=55, col="red")

clust = cutreeStatic(sampleTree, cutHeight = 55, minSize = 10)
table(clust)
#clust

keepSamples = (clust==1)
normalized_counts_t = normalized_counts_t[keepSamples, ]
nGenes = ncol(normalized_counts_t)
nSamples = nrow(normalized_counts_t)
save(normalized_counts_t, file = "dataInput.RData")






### Soft-thresholding    ------  unsigned correlation matrix  (put no difference on positive and negative)
sft <- pickSoftThreshold(normalized_counts_t,
                         dataIsExpr = TRUE,
                         corFnc = cor,
                         networkType = "unsigned"
)

power = sft$powerEstimate


sft_df <- data.frame(sft$fitIndices) %>%
  dplyr::mutate(model_fit = -sign(slope) * SFT.R.sq)


library('ggplot2')

ggplot(sft_df, aes(x = Power, y = model_fit, label = Power)) +
  # Plot the points
  geom_point() +
  # We'll put the Power labels slightly above the data points
  geom_text(nudge_y = 0.1) +
  # We will plot what WGCNA recommends as an R^2 cutoff
  geom_hline(yintercept = 0.80, col = "red") +
  # Just in case our values are low, we want to make sure we can still see the 0.80 level
  ylim(c(min(sft_df$model_fit), 1.05)) +
  # We can add more sensible labels for our axis
  xlab("Soft Threshold (power)") +
  ylab("Scale Free Topology Model Fit, signed R^2") +
  ggtitle("Scale independence") +
  # This adds some nicer aesthetics to our plot
  theme_classic()

powers = c(c(1:10), seq(from = 12, to=20, by=2))
par(mfrow = c(1,2))
cex1 = 0.9
# 横轴是Soft threshold (power)，纵轴是无标度网络的评估参数，数值越高，
# 网络越符合无标度特征 (non-scale)
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)",
     ylab="Scale Free Topology Model Fit,signed R^2",type="n",
     main = paste("Scale independence"))
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     labels=powers,cex=cex1,col="red")
# 筛选标准。R-square=0.8
abline(h=0.8,col="red")


# Soft threshold与平均连通性
plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
     main = paste("Mean connectivity"))
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers,
     cex=cex1, col="red")









### Pick soft power

# nGenes = ncol(normalized_counts_t)
# nSamples = nrow(normalized_counts_t)
# 
# if (is.na(power)){
#   power = ifelse(nSamples<20, ifelse(type == "unsigned", 9, 18),
#                  ifelse(nSamples<30, ifelse(type == "unsigned", 8, 16),
#                         ifelse(nSamples<40, ifelse(type == "unsigned", 7, 14),
#                                ifelse(type == "unsigned", 6, 12))      
#                  )
#   )
# }

power = 7









# # 构建网络
# # deepSplit是生成模块的梯度，从[0:4]中选择，越大模块越多
# # minModuleSize最小模块的基因数目
# # mergeCutHeight是合并相似的模块的合并系数，是通过主成分分析出来的
# # mumericLabels 已数字命名模块
# # nThreads 线程数，当设置为0时使用最大线程

cor <- WGCNA::cor

if(T){
  net = blockwiseModules(
    normalized_counts_t,
    power = power,
    # power = sft$powerEstimate,
    maxBlockSize = 6000,
    TOMType = "unsigned", minModuleSize = 100,
    reassignThreshold = 0, mergeCutHeight = 0.20,
    numericLabels = TRUE, pamRespectsDendro = FALSE,
    saveTOMs = F,
    verbose = 3
  )
  table(net$colors)
}

cor<-stats::cor
# 
# # 查看每个模块的基因数，其中0模块下为没有计算进入模块的基因数
# table(net$colors)
# 
# 
# ####  4

if(T){
  mergedColors = labels2colors(net$colors)
  table(mergedColors)
  moduleColors=mergedColors

  # png("genes-modules.png",width = 800,height = 600)
  # plotDendroAndColors(hierTOM, cbind(dynamicColors, mergedColors),
  #                     c("Dynamic Tree Cut", "Merged dynamic"),
  #                     dendroLabels = FALSE, hang = 0.03,
  #                     addGuide = TRUE, guideHang = 0.05)
  plotDendroAndColors(net$dendrogram[[1]], mergedColors[net$blockGenes[[1]]],
                      "Module colors",
                      dendroLabels = FALSE, hang = 0.03,
                      addGuide = TRUE, guideHang = 0.05)

  # dev.off()
}










Connectivity = softConnectivity(datExpr = normalized_counts_t, power = power) -1
par(mfrow = c(1,1))
scaleFreePlot(Connectivity, main = paste("soft threshold, power = ", power), truncated = F)


ADJ = adjacency(normalized_counts_t, power = power)

TOM = TOMsimilarity(ADJ)
dissTOM = 1 - TOM


### Hierarchical clustering with TOM matrix
hierTOM = hclust(as.dist(dissTOM), method = "average")
par(mfrow = c(1,1))
plot(hierTOM, xlab="", sub="", main = "Gene clustering on TOM-based dissimilarity",
     labels = FALSE, hang = 0.04);

### dynamic tree cutting algorithm
dynamicMods = cutreeDynamic(dendro = hierTOM, distM = dissTOM,
                            deepSplit = 2, pamRespectsDendro = FALSE,
                            minClusterSize = 100);

table(dynamicMods)             ###  0-10: 11modules


#### plot show
dynamicColors = labels2colors(dynamicMods)
table(dynamicColors)

# Plot the dendrogram and colors underneath
sizeGrWindow(8,6)
plotDendroAndColors(hierTOM, dynamicColors, "Dynamic Tree Cut",
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05,
                    main = "Gene dendrogram and module colors")


# Calculate eigengenes
MEList = moduleEigengenes(normalized_counts_t, colors = dynamicColors)
MEs = MEList$eigengenes
# Calculate dissimilarity of module eigengenes
dim(cor(MEs))
MEDiss = 1-cor(MEs);  # 21 samples * 20 modules
# Cluster module eigengenes
METree = hclust(as.dist(MEDiss), method = "average");


par("mar")
par(mar=c(1,5,1,1))
#### Plot the result
plot(METree, main = "Clustering of module eigengenes",
     xlab = "", sub = "")

MEDissThres = 0.2   ###  fuse modules that has 80% correlation
# Plot the cut line into the dendrogram
abline(h=MEDissThres, col = "red")

# Call an automatic merging function
# merge = mergeCloseModules(datExpr[, restConnectivity], dynamicColors, cutHeight = MEDissThres, verbose = 3)
merge = mergeCloseModules(normalized_counts_t, dynamicColors, cutHeight = MEDissThres, verbose = 3)

# The merged module colors
mergedColors = merge$colors;
table(mergedColors)


# Eigengenes of the new merged modules
mergedMEs = merge$newMEs;


### After merging, plot dendrogram         8 modules
plotDendroAndColors(hierTOM, mergedColors, "Tree Cut",
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05,
                    main = "Gene dendrogram and module colors after merging")






library(GEOquery)
a = getGEO('GSE147507')
look_at_a = pData(a[[1]])
group_info = look_at_a[,8]
# look_at_a$treatment = apply(look_at_a,1,group_info)
# test = sapply(metadata, function(x) { x$b <- rep(8,10);return(x)})

metadata=pData(a[[1]])[c(7:9,17:18,21:23,10:12,24:26),][,c(2,3,8)]
# datTraits = data.frame(gsm=rownames(normalized_counts_t),
#                        treatment=trimws(sapply(as.character(metadata$source_name_ch1),function(x) strsplit(x,":")[[1]][2])))
#                        # treatment = rownames(datExpr)
#                        # treatment = metadata[,2])
datTraits = metadata[,c(1,3)]
colnames(datTraits) = c('gsm','treatment')     



sampleNames = rownames(normalized_counts_t);
traitRows = match(sampleNames, sampleNames)  
rownames(datTraits) = datTraits[traitRows, 1] 
library(stringr)
datTraits$treatment = gsub("\\s+", "_", str_trim(datTraits$treatment))
datTraits$treatment <- gsub("-", "_", str_trim(datTraits$treatment))

table(datTraits$treatment)






                       
        

               
#### 5
datExpr = normalized_counts_t
table(datTraits$treatment)
if(T){
  nGenes = ncol(datExpr)
  nSamples = nrow(datExpr)
  
  design=model.matrix(~0+ datTraits$treatment)
  # colnames(design)=levels(datTraits$treatment)
  colnames(design)= c("Mock_treated_A549_cells", "SARS_CoV_2_infected_A549_cells")
  
  # moduleColors <- labels2colors(net$colors)
  moduleColors <- mergedColors
  
  
  # Recalculate MEs with color labels
  MEs0 = moduleEigengenes(datExpr, moduleColors)$eigengenes
  MEs = orderMEs(MEs0); 
  moduleTraitCor = cor(MEs, design , use = "p");
  moduleTraitPvalue = corPvalueStudent(moduleTraitCor, nSamples)
  
  sizeGrWindow(10,6)
  # Will display correlations and their p-values
  textMatrix = paste(signif(moduleTraitCor, 2), "\n(",
                     signif(moduleTraitPvalue, 1), ")", sep = "");
  dim(textMatrix) = dim(moduleTraitCor)
  # png("Module-trait-relationships.png",width = 800,height = 1200,res = 120)
  par(mar = c(8, 7, 3, 3));
  # Display the correlation values within a heatmap plot
  labeledHeatmap(Matrix = moduleTraitCor,
                 xLabels = colnames(design),
                 yLabels = names(MEs),
                 ySymbols = names(MEs),
                 colorLabels = FALSE,
                 colors = blueWhiteRed(50),
                 textMatrix = textMatrix,
                 setStdMargins = FALSE,
                 cex.text = 0.5,
                 zlim = c(-1,1),
                 main = paste("Module-trait relationships"))
  # dev.off()
  
  
  
  ### histogram of gene significance
  SARS_CoV_2_infected = as.data.frame(design[,1]);
  names(SARS_CoV_2_infected) = "SARS_CoV_2_infected"
  y=SARS_CoV_2_infected
  # GS1=as.numeric(cor(y,datExpr[, restConnectivity], use="p"))
  test = cor(y,datExpr, use="p")
  GS1=as.numeric(cor(y,datExpr, use="p"))
  GeneSignificance=abs(GS1)
  # Next module significance is defined as average gene significance.
  ModuleSignificance=tapply(GeneSignificance,
                            moduleColors, mean, na.rm=T)
  sizeGrWindow(8,7)
  # par(mfrow = c(4,1))
  table(moduleColors)
  
  
  plotModuleSignificance(GeneSignificance,moduleColors)
  
}




### Select modules

if(T){
  
  module = "blue"   # 'blue'  20%
  # module = "black"  # 'black'  10%
  # module = "pink"   # 'pink'  10%
  
  
  # Select module probes
  probes = colnames(datExpr) ## 我们例子里面的probe就是基因
  inModule = (moduleColors==module);
  modProbes = probes[inModule];  ########### gene in one module
  
  head(modProbes)  
  length(modProbes)
  # "blue" 1694
  # "black" 149
  # "pink" 1689
  
  
  ############### intra-modular connectivity
  intra_connect = intramodularConnectivity(ADJ, moduleColors, scaleByMax = FALSE)  # ADJ = 16000 x 16000
  
  name_record = list()
  module_intra_record <- matrix(1:length(modProbes), nrow = length(modProbes), ncol = )
  dim(module_intra_record)
  module_intra_record = as.data.frame(module_intra_record);
  for (i in 1:length(modProbes)){
    # print(i)
    probe_this = (rownames(intra_connect) == modProbes[i])
    ind = which(probe_this)
    module_intra_record[i,1] = intra_connect[ind,2]   # (intramodular) connectivity
    name_record = append(name_record,(rownames(intra_connect))[probe_this] ) # (intramodular) genes
  }
  rownames(module_intra_record) = name_record
  
  desc_ord = order(module_intra_record,decreasing = TRUE)
  
  
  module_intra_record_rearra = as.data.frame(module_intra_record[desc_ord,])
  rownames(module_intra_record_rearra) = name_record[desc_ord]
  
  first_ten_percent = round(0.2 * length(module_intra_record_rearra[,1]))
  # first_ten_percent = round(0.1 * length(module_intra_record_rearra[,1]))
  first_ten_name = (rownames(module_intra_record_rearra))[1: first_ten_percent]  # first 10% gene names
  
  first_ten_record <- matrix(1:first_ten_percent, nrow = length(first_ten_percent), ncol = )
  first_ten_record = t(first_ten_record)
  first_ten_record = as.data.frame(first_ten_record);
  rownames(first_ten_record) = first_ten_name
  

  first_ten_record_blue = first_ten_record   # 'blue' 20%  339 genes
  dim(first_ten_record_blue)
  save(first_ten_record_blue,file = 'first_ten_record_blue.Rdata')
  
  
  # first_ten_record_black = first_ten_record   # 'black' 10%  15 genes
  # dim(first_ten_record_black)
  # save(first_ten_record_black,file = 'first_ten_record_black.Rdata')
  
  
  # first_ten_record_pink = first_ten_record   # 'pink' 10%  169 genes
  # dim(first_ten_record_pink)
  # save(first_ten_record_pink,file = 'first_ten_record_pink.Rdata')
  
}

# first_ten_record_pink==first_ten_record_blue



