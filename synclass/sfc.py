"""
SynClass classifier

Date: July, 2022

Developers:
    Brenno Alencar,
    Marcos Vinicius Ferreira
    Ricardo Rios,
    Tatiane Nogueira,
    Tiago Lopes


GNU General Public License v3.0

Permissions of this strong copyleft license are 
	conditioned on making available complete 
	source code of licensed works and 
	modifications, which include larger works 
	using a licensed work, under the same license. 
	Copyright and license notices must be 
	preserved. Contributors provide an express 
	grant of patent rights.
"""
import sys
import numpy as np
import os

import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2.robjects import numpy2ri
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
# sys.path.append('/home/labia/pesquisa/am-pipeline/src')
from ialovecoffe.models import *
from sklearn.metrics import RocCurveDisplay
from collections import Counter
from ialovecoffe.validation import positive_negative_rate
import pandas as pd


def call_wilcox_R (x, y):
    """
    Call R function wilcox.test() to perform wilcox test.

    Parameters
    ----------
    x : numpy.ndarray
        First array of data.
    y : numpy.ndarray
        Second array of data.
        
    Returns
    -------
    pvalue : float
        P-value of the test.

    """
    r.assign("x", x.to_numpy())
    r.assign("y", y.to_numpy())
    r('res<-wilcox.test(x~y)$statistic')
    r_result = r("res")
    return (r_result[0])

def get_statistical_weights(inputSet, labels):
    """
    Get statistical weights for each class.

    Parameters
    ----------
    inputSet : numpy.ndarray
        Input data set.
    ignore_class : float
        Class to ignore.

    Returns
    -------
    weights : numpy.ndarray
        Statistical weights for each class.
        
    """    
    myWeights = np.repeat(np.nan, inputSet.shape[1])

    numpy2ri.activate()    
    for i in np.arange(inputSet.shape[1]):
        myWeights[i] = call_wilcox_R(inputSet.iloc[:,i], labels)

    numpy2ri.deactivate()

    return (1/myWeights)

def auc_eval(y_test, y_pred, positive = 1):
    """
    Calculate AUC per class.

    Parameters
    ----------
    y_test : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.
    positive : int
        Index of positive class.

    Returns
    -------
    auc : float
        Area under the ROC curve.

    """
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=positive)
    return metrics.auc(fpr, tpr)

def accuracy_per_class(y_test, y_pred):
    """
    Calculate accuracy per class.

    Parameters
    ----------
    y_test : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.

    Returns
    -------
    acc_pos: float
        Accuracy for the positive class.
    acc_neg: float
        Accuracy for the negative class.

    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc_pos = tp / (tp + fp)
    acc_neg = tn / (tn + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc_pos, acc_neg,acc

def weighted_mean(x, w):
    """
    Calculate weighted mean.

    Parameters
    ----------
    x : numpy.ndarray
        Array of values.
    w : numpy.ndarray
        Array of weights.

    Returns
    -------
    weighted_mean : float
        Weighted mean.

    """
    return np.dot(x, w)/np.sum(w)

def roc_calc_viz_pred(y_true, y_pred):
    viz = RocCurveDisplay.from_predictions(
                            y_true,
                            y_pred
                        )

    return viz.fpr, viz.tpr, viz.roc_auc


def SynClass(x_train, y_train, x_test, y_test, learner, scoring = 'auc', classThreshold = 0.5, probability = False):
    """
    Perform SynClass.

    Parameters
    ----------
    x_train : numpy.ndarray
        Training data.
    y_train : numpy.ndarray
        Training labels.
    x_test : numpy.ndarray
        Test data.
    y_test : numpy.ndarray
        Test labels.
    learner : str
        Learner to use.
    scoring : str
        Scoring method.
    classThreshold : float
        Class threshold.
    seed : int
        Random seed.
    probability : bool
        Whether to use probability or not.

    Returns
    -------
    acc : float
        Accuracy.        
    auc : float
        Area under the ROC curve.
    f1 : float
        F1 score.
    """
    result_by_att = np.zeros((x_test.shape[0], x_train.shape[1]))

    #fitting a model per attribute
    for att in np.arange(x_train.shape[1]):
        log.debug('-' * 30)
        log.debug(f'Train with {att}')
        log.debug('-' * 30)
        result_by_att[:, att], _, _ = learner(x_train.iloc[:, att], y_train, x_test.iloc[:, att], y_test, prob=probability, metric=scoring)
            

    statistical_weights = get_statistical_weights(x_train, y_train)
    prediction = np.apply_along_axis(weighted_mean, 1, result_by_att, w = statistical_weights)

    y_pred = np.repeat(0, x_test.shape[0])
    y_pred[np.where(prediction > classThreshold)] = 1

    acc_pos, acc_neg = accuracy_per_class(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average='macro')
    auc = auc_eval(y_test, y_pred)
    acc = [acc_pos, acc_neg]

    return acc, f1, auc, prediction, y_pred


def SynClass(x_train, y_train, x_test, y_test, learner, learner_name,scoring = 'auc', classThreshold = 0.5, probability = False, rangeThreshold = [0.1, 0.81, 0.01], results = {},random_state=1):
    """
    Perform SynClass.

    Parameters
    ----------
    x_train : numpy.ndarray
        Training data.
    y_train : numpy.ndarray
        Training labels.
    x_test : numpy.ndarray
        Test data.
    y_test : numpy.ndarray
        Test labels.
    learner : str
        Learner to use.
    scoring : str
        Scoring method.
    classThreshold : float
        Class threshold.
    seed : int
        Random seed.
    probability : bool
        Whether to use probability or not.

    Returns
    -------
    acc : float
        Accuracy.        
    auc : float
        Area under the ROC curve.
    f1 : float
        F1 score.
    """
    result_by_att = np.zeros((x_test.shape[0], x_train.shape[1]))
    #classifiers = dict()

    #fitting a model per attribute'ACC':[],
    # _, _, model= learner(x_train, y_train, x_test, y_test, prob=probability, metric=scoring,random_state=random_state)
    #_,_,model = RSRF(x_train, y_train, x_test, y_test, prob=probability, metric=scoring,random_state=random_state)
    #print("Feature_importances",model.best_estimator_.feature_importances_)
    for att in np.arange(x_train.shape[1]):
        log.debug('-' * 30)
        log.debug(f'Train with {att}')
        log.debug('-' * 30)

        result_by_att[:, att], _, model_cv = learner(x_train.iloc[:, att], y_train, x_test.iloc[:, att], y_test, prob=probability, metric=scoring,random_state=random_state)
        #classifiers[att] = model_cv
    
    # Calcula os pesos dos atributos
    statistical_weights = get_statistical_weights(x_train, y_train)
    # d = pd.DataFrame([model.best_estimator_.feature_importances_,statistical_weights], index = ["FI","Pesos"])

    # statistical_weights = model.best_estimator_.feature_importances_
    prediction = np.apply_along_axis(weighted_mean, 1, result_by_att, w = statistical_weights)
    # print("Pesos",statistical_weights )
   
    # d = d.T

    # # Initialize the MinMaxScaler
    # scaler = MinMaxScaler()

    # # Apply min-max scaling to each row
    # scaled_data = pd.DataFrame(scaler.fit_transform(d), columns=d.columns, index=d.index).round(2)

    # scaled_data.to_csv("feature_importances.csv")

    
    for thresh in np.arange(start=rangeThreshold[0], stop=rangeThreshold[1], step=rangeThreshold[2]):
        log.debug('-' * 30)
        log.debug(f'Threshold with {thresh}')
        

        y_pred = np.repeat(0, x_test.shape[0])
        y_pred[np.where(prediction > thresh)] = 1
        
        acc_pos, acc_neg,acc= accuracy_per_class(y_test, y_pred)

        f1 = f1_score(y_test, y_pred, average='binary')
        auc = auc_eval(y_test, y_pred)
        acc_class = [acc_pos, acc_neg]
        viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, prediction)
        sen,spe,_ = positive_negative_rate(y_test,y_pred)
        log.debug(f'SynClass + {learner_name} F1 ......: {f1}')
        log.debug('-' * 30)

        results['model_name'].append(f'SynClass + {learner_name}')
        results['acc-class-1'].append(acc_pos)
        results['acc-class-2'].append(acc_neg)
        results['ACC'].append(acc)
        results['F1'].append(f1)
        results['ROC'].append(auc)
        results['FPR'].append(viz_fpr)
        results['TPR'].append(viz_tpr)
        results['AUC'].append(viz_auc)
        results['THRE'].append(thresh)
        results['SPE'].append(spe)
        results['SEN'].append(sen)

    return results

def ModelSearchSynClass(x_train, y_train, x_test, y_test, learner, scoring = 'auc', classThreshold = 0.5, probability = False, rangeThreshold = [0.1, 0.81, 0.01], results = {}):
    """
    Perform SynClass.

    Parameters
    ----------
    x_train : numpy.ndarray
        Training data.
    y_train : numpy.ndarray
        Training labels.
    x_test : numpy.ndarray
        Test data.
    y_test : numpy.ndarraydan
        Test labels.
    learner : str
        Learner to use.
    scoring : str
        Scoring method.
    classThreshold : float
        Class threshold.
    seed : int
        Random seed.
    probability : bool
        Whether to use probability or not.

    Returns
    -------
    acc : float
        Accuracy.        
    auc : float
        Area under the ROC curve.
    f1 : float
        F1 score.
    """
    result_by_att = np.zeros((x_test.shape[0], x_train.shape[1]))

    #classifiers = dict()

    #fitting a model per attribute'ACC':[],
    for att in np.arange(x_train.shape[1]):
        dic_models = {0: RSIsolationForest, 1: RSLocalOutlierFactor_ml, 2: RSLocalOutlierFactor_ml,
                      3: RSIsolationForest, 4: RSIsolationForest, 5: RSOneClassSVM, 6: RSOneClassSVM}
        learner = dic_models[att]

        log.debug('-' * 30)
        log.debug(f'Train with {att} - Learner {learner}')
        log.debug('-' * 30)

        x_train_reshape = x_train.iloc[:, att].values.reshape(-1, 1)
        x_test_reshape = x_test.iloc[:, att].values.reshape(-1, 1)
        result_by_att[:, att], _, _ = learner(x_train_reshape, y_train, x_test_reshape, y_test, prob=probability, metric=scoring)
        #classifiers[att] = model_cv

    for thresh in np.arange(start=rangeThreshold[0], stop=rangeThreshold[1], step=rangeThreshold[2]):
        log.debug('-' * 30)
        log.debug(f'Threshold with {thresh}')
        
        statistical_weights = get_statistical_weights(x_train, y_train)
        prediction = np.apply_along_axis(weighted_mean, 1, result_by_att, w = statistical_weights)

        y_pred = np.repeat(0, x_test.shape[0])
        y_pred[np.where(prediction > thresh)] = 1
        
        acc_pos, acc_neg = accuracy_per_class(y_test, y_pred)

        f1 = f1_score(y_test, y_pred, average='macro')
        auc = auc_eval(y_test, y_pred)
        acc = [acc_pos, acc_neg]
        viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, prediction)

        log.debug(f'SynClass F1 ......: {f1}')
        log.debug('-' * 30)

        results['model_name'].append('SynClass')
        results['acc-class-1'].append(acc_pos)
        results['acc-class-2'].append(acc_neg)
        results['F1'].append(f1)
        results['ROC'].append(auc)
        results['FPR'].append(viz_fpr)
        results['TPR'].append(viz_tpr)
        results['AUC'].append(viz_auc)
        results['THRE'].append(thresh)

    return results