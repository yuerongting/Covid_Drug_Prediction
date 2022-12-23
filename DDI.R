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


# DDI only
library(dnet)
load("DTI_30_9.Rdata")
load("DTI_known.Rdata")
DTI
load("drug_interact.Rdata") ###   DDI:  340 drugs
test  
drug_interact <- test ###  
length(unique(drug_interact$parent_key))  ###  DDI:  340 drugs


load("drug_interact_all.Rdata")
drug_interaction 
drug_interact <- drug_interaction ### All DDI: 3366 drugs
length(unique(drug_interact$parent_key))


# DDI_all <- graph_from_data_frame(drug_interact, directed = FALSE) ### 

load('drug_SMILES_468.Rdata')
drug_SMILES  ### 468 drugs SMILES
length(unique(drug_SMILES$parent_key))  ### 468 drugs SMILES

load('drug_common.Rdata')
drug_common  ### 214 drugs from DTI


# load('drug_SMILES.Rdata')
# drug_SMILES  


# drug_interact <- filter(test, parent_key %in% rownames(DTI) & `drugbank-id` %in% rownames(DTI))  # 30 drugs

# DDI: 468 drugs, 48460 edges
drug_interact <- filter(drug_interact, parent_key %in% drug_SMILES$parent_key & `drugbank-id` %in% drug_SMILES$parent_key)  # drug SMILES
drug_interact_common <- filter(drug_interact, parent_key %in% drug_common & `drugbank-id` %in% drug_common)   # DDI: drug in common


##### Save DDI
# DDI_dat_frame <- filter(drug_interact, `drugbank-id` %in% drug_common)
# DDI_dat_frame <- filter(DDI_dat_frame, parent_key %in% drug_common)
# write.csv(DDI_dat_frame,'c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation\\DDI_dat_frame.csv')

write.csv(drug_SMILES,'c:\\Users\\yrt05\\Desktop\\Covid_coference_presentation\\DDI_dat_frame.csv')







drug_interact_1 <- filter(drug_interact, `drugbank-id` %in% drug_common )
drug_interact_2 <- filter(drug_interact, parent_key %in% drug_common )

unique(drug_interact_1$`drugbank-id`)
unique(drug_interact_2$`drugbank-id`)

intersect(union(unique(drug_interact_1$`drugbank-id`), unique(drug_interact_2$`drugbank-id`)),drug_common)



setdiff(drug_interact$parent_key, drug_common)

# DDI: 340  out of 468 drugs, 17802 edges
DDI <- graph_from_data_frame(drug_interact, directed = FALSE)  

length(unique(as_ids(V(DDI))))
setdiff(as_ids(V(DDI)), drug_common)
setdiff( drug_common, as_ids(V(DDI)))




# save(DDI, file = "DDI.Rdata")   ### 340  out of 468 drugs, have 48460 edges
# save(DDI, file = "DDI_all.Rdata")   ### All DDI ,  3366 drugs for all distribution
unique(drug_interact$`drugbank-id`)

load("DDI.Rdata")
DDI

# union(drug_interact$`drugbank-id`, drug_interact$parent_key)
# filter(drug_interact, `drugbank-id` %in% drug_interact$parent_key)
unique(drug_interact$parent_key)








# Drug Chemical Data ---------------------------------------------------------------

Download_data = FALSE
library("dbparser")

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
  drug_groups <- drug_groups()
  
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


druggs <- drugs

### filter chemical drugs (not biotech)
chemical_drugs_index <- which(druggs[["general_information"]][["type"]] == "small molecule")
chemical_drugs_name <- (drugs[["general_information"]][["primary_key"]])[chemical_drugs_index]

# save(chemical_drugs_name, file = "chemical_drugs_name.Rdata")   
load("chemical_drugs_name.Rdata")
chemical_drugs_name

load('drug_SMILES_468.Rdata')
drug_SMILES
drug_name <- drug_SMILES$parent_key



########### Drug data property---------------------------------------------------------------

drug_data <- filter(drug_calc_prop, parent_key %in% drug_name)

item_name <- drug_data[c(1:26),1]

x = matrix(, nrow = length(drug_name), ncol = 20)  # 4 + 2 + 14
colnames(x) <- item_name$kind[c(1:4, 7:8, 13:26)]
rownames(x) <- drug_name
for (j in 1:length(drug_name))  # length 468
{
  for (i in 1:(nrow(drug_data)) )  # length 11997
  {
    if (drug_data$parent_key[i]==drug_name[j] )
    {
      if(drug_data$kind[i]=='logP' && drug_data$source[i]=='ALOGPS')
        {
        x[j,1] <- drug_data$value[i]
      }
      else 
        if(drug_data$kind[i]=='logS' && drug_data$source[i]=='ALOGPS')
        {
          x[j,2] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]=='Water Solubility' && drug_data$source[i]=='ALOGPS')
        {
          Solubility <- gsub(" g/l","",drug_data$value[i])
          x[j,3] <- Solubility
        }
      else
        if(drug_data$kind[i]=='logP' && drug_data$source[i]=='ChemAxon')
        {
          x[j,4] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]=='Molecular Weight' && drug_data$source[i]=='ChemAxon')
        {
          x[j,5] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]=='Monoisotopic Weight' && drug_data$source[i]=='ChemAxon')
        {
          x[j,6] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]=='Polar Surface Area (PSA)' && drug_data$source[i]=='ChemAxon')
        {
          x[j,7] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]=='Refractivity' && drug_data$source[i]=='ChemAxon')
        {
          x[j,8] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]=='Polarizability' && drug_data$source[i]=='ChemAxon')
        {
          x[j,9] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[16] && drug_data$source[i]=='ChemAxon')
        {
          x[j,10] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[17] && drug_data$source[i]=='ChemAxon')
        {
          x[j,11] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[18] && drug_data$source[i]=='ChemAxon')
        {
          x[j,12] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[19] && drug_data$source[i]=='ChemAxon')
        {
          x[j,13] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[20] && drug_data$source[i]=='ChemAxon')
        {
          x[j,14] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[21] && drug_data$source[i]=='ChemAxon')
        {
          x[j,15] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[22] && drug_data$source[i]=='ChemAxon')
        {
          x[j,16] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[23] && drug_data$source[i]=='ChemAxon')
        {
          x[j,17] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[24] && drug_data$source[i]=='ChemAxon')
        {
          x[j,18] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[25] && drug_data$source[i]=='ChemAxon')
        {
          x[j,19] <- drug_data$value[i]
        }
      else
        if(drug_data$kind[i]==item_name$kind[26] && drug_data$source[i]=='ChemAxon')
        {
          x[j,20] <- drug_data$value[i]
        }
    }
  }
}

x[is.na(x)] <- 0
x <- matrix(as.numeric(x),    # Convert to numeric matrix
                  ncol = ncol(x))

library(BBmisc)
x <- normalize(x, method = "range", margin = 2, range = c(0, 1))

# load('drugs.Rdata')
# drugs <- drugs$general_information
# drugs_data <- drugs %>% as_tibble() %>% dplyr::select(primary_key, name, type, state, average_mass, monoisotopic_mass)
# colnames(drugs_data)[2] <- "drug_name"
# drugs_data_property <- filter(drugs_data, primary_key)


# save(x, file = 'x_drug.Rdata')




# Drug Data ---------------------------------------------------------------
### DDI
###
library(Rcpp)
library(readxl)
# drug_data <- read_excel("Drug_data.xlsx")
load("drug_interaction.Rdata")  
drug_interaction <- drug_interaction[, c(1,4)]    ### All drug interaction

load('drug_SMILES.Rdata') ## All SMILES drug
drug_SMILES_all
drug_interaction <- filter(drug_interaction, `drugbank-id` %in% drug_SMILES_all$parent_key)
drug_interaction <- filter(drug_interaction, parent_key %in% drug_SMILES_all$parent_key)  ### All drug SMILES with interactions

### Find all drugs that can interact with these 214 drugs
load('drug_common.Rdata')  
drug_id <- drug_common  ### Common 214 drugs for DTI

drug_interact_1 <- drug_interaction[which(drug_interaction[,1] == drug_id), ]  
drug_interact_2 <- drug_interaction[which(drug_interaction[,2] == drug_id), ] 
library(dplyr)
drug_name_no_repeat <- (bind_rows(drug_interact_1[,1], drug_interact_1[,2], drug_interact_2[,1], drug_interact_2[,2] ))[,1]

drug_id_tib <- tibble(x = drug_id)


colnames(drug_id_tib) <- colnames(drug_name_no_repeat)
drug_name_no_repeat <- bind_rows(  drug_name_no_repeat[,1], drug_id_tib[,1]  )

drug_name <- drug_name_no_repeat[!duplicated(drug_name_no_repeat),] 
drug_name <- na.omit(drug_name) 

load("chemical_drugs_name.Rdata")
drug_name <- drug_name %>% filter(`drugbank-id` %in% chemical_drugs_name)


setdiff(drug_id, drug_name$`drugbank-id`)


# save(drug_name, file = "drug_name.Rdata")           ### Find DDI 468 drugs for DDI (include original 214 drugs)
# write(as.matrix(drug_name),"drug_name.txt",sep = '') 

test <- drug_interaction %>% filter(parent_key %in% drug_name$`drugbank-id` & `drugbank-id` %in% drug_name$`drugbank-id` )
# unique(drug_interaction$`drugbank-id`)    ### All drug names in all drug interaction

# save(test, file = "drug_interact.Rdata")     ### DDI has 48450 edges
# save(drug_interaction, file = "drug_interact_all.Rdata")     ### All DDI
