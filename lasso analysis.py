import pandas, os
from sklearn import model_selection
from sklearn.linear_model import Lasso, LassoCV
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import auc

def roc_curve(y_true, y_prob, thresholds):
    fpr = []
    tpr = []

    for threshold in thresholds:
        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]


def getLassoFeatures(featuredict, featurelist, coeflist):
    
    """
    Updates featuredict object to examine LASSO coefficients and covariables
    for examination of model feature selection.
    
    
    Parameters
    ----------
    
    featuredict : dict
        Dictionary where keys are column titles in feature matrix. Each key
    corresponds to a dictionary with the following basic structure:
        
        'coef_set' : list
            list of coefficient values over nested lasso history
        
        'covariables' : list 
            set of covariables selected with this feature by lasso
        
        'frequency' : int
            frequency this feature was selected by LASSO model as important
            
    featurelist : list
        list of column names of non-zero coefficient entries from LASSO model
    
    coeflist : list
        list of non-zero coefficient entries from LASSO model
    
    
    Returns
    -------
    
    featuredict : dict
        updated feature dictionary
    
    """
    
    for feat, coef in zip(featurelist, coeflist):
        featuredict[feat]['coef_set'].append(coef)
        featuredict[feat]['covariables'].append([f for f in featurelist if f != feat])
        featuredict[feat]['frequency'] += 1
    
    return featuredict


def nestedCVLassoRegression(x, 
                            y, 
                            nFold, 
                            featuredict,
                            shuffle=True, 
                            CaseGroup=True):
    scaler = preprocessing.StandardScaler()

    groupkfold = model_selection.StratifiedGroupKFold(n_splits=nFold, shuffle=shuffle)

    mean_auc = []
    mean_fpr = []
    mean_tpr = []
    
    if CaseGroup:
        groups = x.Case_id
    else:
        groups = x.Biopsy_id

    for i, (train, test) in enumerate(groupkfold.split(x, y, groups=groups)):

        xfin = dropbyname(x,['Biopsy','Case'])

        xfin = xfin.to_numpy()
        xtrain, ytrain = xfin[train, :], y[train]
        xval, yval = xfin[test, :], y[test]

        xtrain = scaler.fit_transform(xtrain)
        xval = scaler.transform(xval)

        try:
            #find optimal alpha value
            estimator = LassoCV(cv=nFold, max_iter=1500000, tol=0.01).fit(xtrain, ytrain)
            estimator = Lasso(estimator.alpha_) #fit lasso using the optimal alpha
            estimator.fit(xtrain, ytrain)
            featurelist = list(x.columns[2:][np.where(np.abs(estimator.coef_)>0)])
            coeflist = list(estimator.coef_[np.where(np.abs(estimator.coef_)>0)])
            featuredict = getLassoFeatures(featuredict, featurelist, coeflist)

            y_prob_val = estimator.predict(xval)

            ths = np.linspace(0, 1, num=11)

            fpr, tpr = roc_curve(yval, y_prob_val, ths)


            mean_fpr.append([fpr])
            mean_tpr.append([tpr])
            mean_auc.append(auc(fpr, tpr))
        except:
            ...

    std_aucs = np.std(mean_auc)
    mean_auc = np.mean(mean_auc)
    mean_fpr = np.mean(mean_fpr, axis=0)
    mean_tpr = np.mean(mean_tpr, axis=0)
    return mean_auc, mean_fpr, mean_tpr, featuredict


def runNestedCVMultipleTimes(x, y, k, nIter, shuffle=True, CaseGroup=True):
    mean_aucs = []
    mean_fprs = []
    mean_tprs = []
    featuredict = {feat:{'coef_set' : [],
                     'covariables' : [],
                     'frequency' : 0,
                    } for feat in x.columns[2:]}

    for i in tqdm(range(nIter)):
        mean_auc, mean_fpr, mean_tpr, featuredict = nestedCVLassoRegression(x, 
                                                               y, 
                                                               k, 
                                                               featuredict,
                                                               shuffle=shuffle, 
                                                               CaseGroup=CaseGroup)

        mean_aucs.append(mean_auc)
        mean_fprs.append(mean_fpr)
        mean_tprs.append(mean_tpr)
    print(mean_fpr.shape)
    std_aucs = np.std(mean_aucs)
    mean_aucs = np.mean(mean_aucs)
    mean_fprs = np.mean(mean_fprs, axis=0)[0, :]
    mean_tprs = np.mean(mean_tprs, axis=0)[0, :]
    return mean_aucs, std_aucs, mean_tprs, mean_fprs, featuredict