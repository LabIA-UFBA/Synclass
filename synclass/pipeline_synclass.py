import sys
import time
import pickle
import warnings
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger as log
from datetime import datetime
from collections import Counter
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from synclass.sfc import SynClass
from collections import Counter


# from experiments.experiment_bracis_2023 import undersampling
from ialovecoffe.validation import computer_scores, computer_scores_outlier, accuracy_per_class
from ialovecoffe.models import RSDT_synclass
from sklearn.utils import shuffle
from sklearn.metrics import RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier

# config
warnings.simplefilter("ignore")
random.seed(10)

def undersampling(X, Y, percentage, rs, at='target', increase=1, sm=True):
    
    X[at] = Y
    
    # surffle
    X = shuffle(X, random_state=rs)

    #size_minority = min(Counter(X[at]).values())
    proportions = Counter(X[at])

    class_minority = min(proportions, key=proportions.get)
    size_minority  = proportions[class_minority]
    
    p = np.ceil(size_minority * percentage).astype('int')
    p_train = (size_minority - p)
        
    train, test = [], []

    for classe in X[at].unique():
        
        df_class = X[X[at] == classe]

        if classe != class_minority:
            train.append(df_class.iloc[p:(p_train*increase)])
        else:
            train.append(df_class.iloc[p:(p_train)])        
            
        test.append(df_class.iloc[:p])
        #train.append(df_class.iloc[p:p_train])
        
    df_train = pd.concat(train)
    df_test = pd.concat(test)
    
    y_train = df_train[at]
    y_test = df_test[at]
        
    x_train = df_train.drop([at], axis=1)
    x_test = df_test.drop([at], axis=1)   

    if sm:
        x_train, y_train = SMOTE(random_state=rs).fit_resample(x_train, y_train)
    
    return x_train, y_train, x_test, y_test

def format_metric_with_std(row, metric):
    value = row[metric]
    std = row[f"{metric}_std"]
    return f"{value} +/- {std}"


def get_table_with_std(df_folds,df_describe):
  
    # folds  = pd.read_csv(f'{exp_name}/Raw_Results/{filename}_{test_size}.csv')
    # df_describe = pd.read_csv(f'{exp_name}/Describe_{filename}_{test_size}.csv')
    by = ["model_name"]
    desired_metrics = ["F1","acc-class-1","acc-class-2","AUC", "SEN","SPE"]
    # folds_std = df_folds.groupby("model_name")[["F1","acc-class-1","acc-class-2","AUC","SEN","SPE"]].std().round(2)
    folds_std = df_folds.groupby(by="model_name")[desired_metrics].std().round(2)
    columns = folds_std.columns

    std_columns = {col: f"{col}_std" for col in columns if col != 'model_name'}

    folds_std.rename(columns=std_columns, inplace=True)

    df_merged = pd.merge(df_describe,folds_std,on=by)
    

    for metric in columns:
        df_merged[metric] = df_merged.apply(format_metric_with_std,metric=metric,axis=1)
    

    df_final = df_merged.drop(columns=[f"{metric}_std" for metric in desired_metrics] )
    for col in df_final.columns:
        if col in desired_metrics or col in by:
            continue
            #df_final.drop(columns=["THRE","ACC","ROC"],inplace=True)
        df_final.drop(col, inplace=True, axis = 1)

    return df_final   
    # df_final.to_csv(f'out/synclass/{exp_name}/std_tables/{filename}_{test_size}_with_std.csv',index=False)
    # df_final.to_latex(f'{exp_name}/latex_tables/{filename}_{test_size}_latex.tex', index=False)

def get_best_thresh_metrics(df,target_metric,metrics={
    "F1":"mean",
    "ROC":"mean",
    "acc-class-1":"mean",
    "acc-class-2":"mean",
    "ACC":"mean",
    "AUC":"mean",
    "SEN":"mean",
    "SPE":"mean"}):

    # Calculate mean accuracy and F1 for each threshold
    mean_values = df.groupby(["model_name",'THRE']).agg(metrics).reset_index()

    best_target_metric = mean_values.loc[mean_values.groupby("model_name")[target_metric].idxmax()]
    best_target_metric = best_target_metric.round(2)
    return best_target_metric

def test_balacing(X, Y, percentage, rs, at='target', sm=False):
    
    X[at] = Y

    size_minority = min(Counter(X[at]).values())
    
    p = np.ceil(size_minority * percentage).astype('int')
    train = []
    test = []
    for classe in X[at].unique():
        
        df_class = X[X[at] == classe]
        
        test.append(df_class.iloc[:p])
        train.append(df_class.iloc[p:])
        
    df_train = pd.concat(train)
    df_test = pd.concat(test)

    # surffle
    df_train = shuffle(df_train, random_state=rs)
    df_test = shuffle(df_test, random_state=rs)
    
    y_train = df_train[at]
    y_test = df_test[at]
        
    x_train = df_train.drop([at], axis=1)
    x_test = df_test.drop([at], axis=1)   

    if sm:
        x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    
    return x_train, y_train, x_test, y_test

def test_balacing_random(X, Y, percentage, rs, at='target', sm=False):
    
    # surffle
    X = shuffle(X, random_state=rs)
    Y = shuffle(Y, random_state=rs)

    X[at] = Y
    size_minority = min(Counter(X[at]).values())
    
    p = np.ceil(size_minority * percentage).astype('int')
    train = []
    test = []
    for classe in X[at].unique():
        
        df_class = X[X[at] == classe]
        
        test.append(df_class.iloc[:p])
        train.append(df_class.iloc[p:])
        
    df_train = pd.concat(train)
    df_test = pd.concat(test)
      
    y_train = df_train[at]
    y_test = df_test[at]
        
    x_train = df_train.drop([at], axis=1)
    x_test = df_test.drop([at], axis=1)   

    if sm:
        x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    
    return x_train, y_train, x_test, y_test


def roc_calc_viz(classifier, X_test, y_test):
    viz = RocCurveDisplay.from_estimator(
                classifier,
                X_test,
                y_test
            )
    
    return viz.fpr, viz.tpr, viz.roc_auc

def roc_calc_viz_pred(y_true, y_pred):
    viz = RocCurveDisplay.from_predictions(
                            y_true,
                            y_pred
                        )

    return viz.fpr, viz.tpr, viz.roc_auc


def run_experiment(x, y, iterations, p, dsetname,learners = [("DT",RSDT_synclass)],smote=False) -> pd.DataFrame:

    data_results = []

    #results = {'model_name': [], 'iteration':[], 'F1':[], 
    #            'ROC':[],'acc-class-1':[],'acc-class-2':[], 'SEN':[], 
    #            'SPE':[], 'MCC':[], 'TPR': [], 'FPR':[], 'AUC': [], 'THRE': []}

    final_results = {'model_name': [], 'F1':[],   'ROC':[], 'acc-class-1':[], 'acc-class-2':[], "ACC":[],
                    'TPR': [], 'FPR':[], 'AUC': [], 'THRE': [], 'SEN': [], 'SPE': []}

    for i in tqdm(range(iterations)):
        
        for (learner_name,learner) in learners:
            #x_train_raw, y_train_raw, x_test_raw, y_test_raw = test_balacing(x, y, p, i, False)
            #x_train_raw, y_train_raw, x_test_raw, y_test_raw = test_balacing_random(x, y, p, i)
            x_train_raw, y_train_raw, x_test_raw, y_test_raw = undersampling(x, y, p,sm=smote, rs = i)

            # Feature Selection

            # clf = DecisionTreeClassifier(max_depth=16,random_state=i)
            # clf.fit(x_train_raw, y_train_raw)

            # importances = clf.feature_importances_
            # print(importances)
            # x.drop("target",inplace=True,axis=1)
            # print(x)
            # selected_features = x.columns[importances > 0.1]
            # print(selected_features)
            # x_train_raw = x_train_raw[selected_features] 
            # x_test_raw = x_test_raw[selected_features]

            #print(x_train_raw)
            #print(Counter(y_train_raw))
            
            #x_train = x_train_raw.to_numpy()
            #x_test = x_test_raw.to_numpy()
            #y_train = y_train_raw.to_numpy()
            #y_test = y_test_raw.to_numpy()

            log.debug('-' * 30)
            log.debug(f'{dsetname} - v2 - Iteration {i} - test size: {p}')
            log.debug('-' * 30)

            results_iter = SynClass(x_train_raw, y_train_raw, x_test_raw, y_test_raw, learner = learner, learner_name=learner_name,scoring = 'f1', classThreshold = 0.5, probability = False, rangeThreshold = [0.1, 0.95, 0.01], results = final_results,random_state=i)

            log.debug('\n')


    df_fold = pd.DataFrame(final_results)
    models = df_fold['model_name'].unique()
    log.info('\n')
    log.info('-' * 30)
    for model in models:

        df_model = df_fold[df_fold['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')

        log.info(f'MODEL {model} with .....: {mean_f1}')

    return df_fold
 