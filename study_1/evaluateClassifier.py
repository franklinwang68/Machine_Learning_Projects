# -*- coding: utf-8 -*-
"""
Script used to evaluate classifier accuracy

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from classifySpam import predictTest

desiredFPR = 0.01
trainDataFilename = 'spamTrain1.csv'
testDataFilename = 'spamTrain2.csv'
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

def evalModel(models):
    data1 = np.loadtxt(trainDataFilename,delimiter=',')
    data2 = np.loadtxt(testDataFilename,delimiter=',')

    data = np.r_[data1,data2]
    X = data[:, :-1]
    y = data[:, -1]

    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

    for model in models:
        testOutputs = model.predictTest(trainFeatures,trainLabels,testFeatures)
        aucTestRun = roc_auc_score(testLabels,testOutputs)
        tprAtDesiredFPR,fpr,tpr = tprAtFPR(testLabels,testOutputs,desiredFPR)

        plt.plot(fpr,tpr, label=model)

        print(f'{model} | AUC: {aucTestRun} | TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for spam detector')
    plt.legend()
    plt.show()

