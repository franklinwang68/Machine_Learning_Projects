# -*- coding: utf-8 -*-

### TODO MUST PIP INSTALL xgboost

from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

"""
Use an XGBoost model 
"""
def predictTest(trainFeatures,trainLabels,testFeatures):
    model = xgb.XGBClassifier(n_estimators=15, gamma=1, max_depth = 2, eval_metric='logloss', random_state=1, alpha = 0.1)

    imputer = SimpleImputer(missing_values=-1, strategy='mean')
    trainFeatures = imputer.fit_transform(trainFeatures)
    testFeatures = imputer.transform(testFeatures)

    scaler = StandardScaler()
    trainFeatures = scaler.fit_transform(trainFeatures)
    testFeatures = scaler.transform(testFeatures)

    # Use Recursive Feature Elimination to select features
    # 15 chosen by maximizing auc and tpr over 50 different splits of training / test data.
    n_features_to_select = 15
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe = rfe.fit(trainFeatures, trainLabels)

    trainFeatures = rfe.transform(trainFeatures)
    testFeatures = rfe.transform(testFeatures)

    # Train the model on selected features
    model.fit(trainFeatures, trainLabels)

    testOutputs = model.predict_proba(testFeatures)[:,1]

    return testOutputs
