## protein of interest (POI)
load("drug_target_inter.Rdata")
drug_target_inter


load("gene_name.Rdata")
gene_name

# protein = c("TLR2","ITGB3","CFP","PIK3CA","BAX","CCL5","IL6","IL1B","PTPN11")
protein = gene_name  ### 495 genes

# save(protein, file = "protein_POI.Rdata")


### Protein property ---------------------------------------
# install.packages("UniprotR")
library(UniprotR) 








drug_target_inter_data <- filter(drug_target_inter, gene_name %in% protein)   

### 295 drugs for 7 targets
drug_nam <- drug_target_inter_data[,1]
drug_nam <- unique(drug_nam)
drug_nam <- as.matrix(drug_nam)

# drug_data <- read_excel("Drug_data.xlsx")
# drug_id_37 <- drug_data[1:56,] 
# drug_id_37 <- drug_id_37[-(which(drug_id_37$Biotech == 1)),]
# drug_id_37 <- drug_id_37$`Drug_bank`
# drug_id_37 <- drug_id_37[!duplicated(drug_id_37)]
# drug_common <- intersect(drug_nam, drug_id_37)


save(drug_common, file = 'drug_common.Rdata')   # 35 drugs in common (manual & database)