# -*- coding: utf-8 -*-
"""
Script used to evaluate classifier accuracy

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
from classifySpam import predictTest

desiredFPR = 0.01
train1DataFilename = 'spamTrain1.csv'
train2DataFilename = 'spamTrain2.csv'
#testDataFilename = 'spamTest.csv'

def tprAtFPR(labels,outputs,desiredFPR):
    fpr,tpr,thres = roc_curve(labels,outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr<=desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex+1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex+1]
    tprAt = ((tprAbove-tprBelow)/(fprAbove-fprBelow)*(desiredFPR-fprBelow) 
             + tprBelow)
    return tprAt,fpr,tpr

train1Data = np.loadtxt(train1DataFilename,delimiter=',')
#train2Data = np.loadtxt(train2DataFilename,delimiter=',')
#trainData = np.r_[train1Data,train2Data]
trainData = train1Data
testData = np.loadtxt(train2DataFilename,delimiter=',')

# Randomly shuffle rows of training and test sets then separate labels
# (last column)
shuffleIndex = np.arange(np.shape(trainData)[0])
np.random.shuffle(shuffleIndex)
trainData = trainData[shuffleIndex,:]
trainFeatures = trainData[:,:-1]
trainLabels = trainData[:,-1]

shuffleIndex = np.arange(np.shape(testData)[0])
np.random.shuffle(shuffleIndex)
testData = testData[shuffleIndex,:]
testFeatures = testData[:,:-1]
testLabels = testData[:,-1]

testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
aucTestRun = roc_auc_score(testLabels,testOutputs)
tprAtDesiredFPR,fpr,tpr = tprAtFPR(testLabels,testOutputs,desiredFPR)

plt.plot(fpr,tpr)

print(f'Test set AUC: {aucTestRun}')
print(f'TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for spam detector')    
plt.show()
