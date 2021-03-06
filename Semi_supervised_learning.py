# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:43:29 2021

@author: yrt05
"""


cd "C:\Users\yrt05\Desktop\Covid_coference_presentation"

import pyreadr
import sklearn 

prot_simi = (pyreadr.read_r('psimmat.RData')['psimmat']).values # protein similarity


drug_simi = (pyreadr.read_r('jaccard_sim.Rdata')['jaccard_sim']).values # drug similarity


DTI = (pyreadr.read_r('drug_target_inter_data.Rdata')['drug_target_inter_data']).values # DTI

import csv




# tsv_file = open("DTI_database.tsv")
# read_tsv = csv.reader(tsv_file, delimiter="\t")
# for row in read_tsv:
#     print(row)

import pandas as pd

df = pd.read_csv("DTI_database.tsv", sep='\t')




cd "C:\Users\yrt05\Desktop\semisup-learn-master\semisup-learn-master"
### Example
import numpy as np
import random
from frameworks.CPLELearning import CPLELearningModel
from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import sklearn.svm
from methods.scikitWQDA import WQDA
from frameworks.SelfLearning import SelfLearningModel

# load data
cancer = fetch_mldata("Lung cancer (Ontario)")
X = cancer.target.T
ytrue = np.copy(cancer.data).flatten()
ytrue[ytrue>0]=1

# label a few points 
labeled_N = 4
ys = np.array([-1]*len(ytrue)) # -1 denotes unlabeled point
random_labeled_points = random.sample(np.where(ytrue == 0)[0], labeled_N/2)+\
                        random.sample(np.where(ytrue == 1)[0], labeled_N/2)
ys[random_labeled_points] = ytrue[random_labeled_points]

# supervised score 
#basemodel = WQDA() # weighted Quadratic Discriminant Analysis
basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])
print "supervised log.reg. score", basemodel.score(X, ytrue)

# fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X, ys)
print "self-learning log.reg. score", ssmodel.score(X, ytrue)

# semi-supervised score (base model has to be able to take weighted samples)
ssmodel = CPLELearningModel(basemodel)
ssmodel.fit(X, ys)
print "CPLE semi-supervised log.reg. score", ssmodel.score(X, ytrue)

# semi-supervised score, WQDA model
ssmodel = CPLELearningModel(WQDA(), predict_from_probabilities=True) # weighted Quadratic Discriminant Analysis
ssmodel.fit(X, ys)
print "CPLE semi-supervised WQDA score", ssmodel.score(X, ytrue)

# semi-supervised score, RBF SVM model
ssmodel = CPLELearningModel(sklearn.svm.SVC(kernel="rbf", probability=True), predict_from_probabilities=True) # RBF SVM
ssmodel.fit(X, ys)
print "CPLE semi-supervised RBF SVM score", ssmodel.score(X, ytrue)








# ### Given drugs "identifiers", find chemical structure   
# from urllib.request import urlopen
# from urllib.parse import quote

# def CIRconvert(ids):
#     try:
#         url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
#         ans = urlopen(url).read().decode('utf8')
#         return ans
#     except:
#         return 'Did not work'

# identifiers  = ['3-Methylheptane', 'Aspirin', 'Diethylsulfate', 'Diethyl sulfate', '50-78-2', 'Adamant']

# for ids in identifiers :
#     print(ids, CIRconvert(ids))