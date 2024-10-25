# io, system, data
import os
import sys
import time
import pickle
import random
import pandas as pd
import numpy as np
from loguru import logger as log
from warnings import simplefilter
from datetime import datetime
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

# preprocessing
from ialovecoffe.data import kmeans_split_data, pre_processing, data_splitting
from ialovecoffe.data import under_sampling_equals_class, kmeans_split_data

# Feature selection
from sklearn.feature_selection import *

# models
from ialovecoffe.models import *

# metrics - validation
from ialovecoffe.validation import computer_scores,computer_scores_roc

from experiment_synclass import pre_processing_hemoA,pre_processing_hemoB,pre_processing_rin,get_table_with_std
from synclass.pipeline_synclass import undersampling
 
import warnings

warnings.filterwarnings("ignore")

def run_all_pipeline_under_smote_techiques(experiment_info,folder,test_size=0.2):
    '''
    Run pipile cross validation with iterations
    Option II - 15 yes and 15 no (random selection)
    '''
    random.seed(10)
    n_iter = 1
    NF = 5
    N_JOBS = -1

    results = {'scaling_method':[],
               'iteration':[], 
               #'fold':[], 
               'oversampling_technique': [],
               'model_name': [],
               #'ACC':[],
               'acc-class-1':[],
               'acc-class-2':[],
               'TPR':[],
               'FPR':[],
               'AUC':[],
               'F1':[], 
               'ROC':[],
            #    'IOU':[], 
            #    'FMI':[], 
            #    'MCC':[], 
               'SEN':[], 
               'SPE':[]}

    oversampling_techniques = {
        "NoSMOTE": lambda x_train,y_train: (x_train,y_train),
    #     "SMOTE":over_sampling_smote,
    #     "SMOTETEK":over_sampling_smote_tek,
    #    "BorderlineSMOTE":over_sampling_borderline_smote,
    #     "SMOTEENN": over_sampling_smote_enn,
    #     "ADASYN":over_sampling_adasyn
    }

    os.makedirs(f'out/{folder}/Raw_Results',exist_ok=True)

    full_path = experiment_info["dataset_info"]["dataset_path"] +experiment_info["dataset_info"]["dsetname"] + '.csv'
    dataset = pd.read_csv(full_path,sep = experiment_info["dataset_info"]["dataset_sep"])
    
    x,y,dataset = experiment_info["pre_processing_func"](dataset)
    # x = x.to_numpy()
    # y = y.to_numpy()

    for scaling in ["NoScaling"]:
        # pre processed data
        # x, y, df = pre_processing(df,target_feature="OM", scaling=scaling)
        
        for i in range(1,n_iter+1):
            
            # troca de samples
          
            
            ini_j = time.time()
            log.info(f'iter: {i}')
            for oversampling_technique, oversample_data in oversampling_techniques.items():
            
                # x_train,y_train = oversample_data(x[train_index],y[train_index])
                # x_test,y_test = x[test_index],y[test_index]
                x_train_raw, y_train_raw, x_test_raw, y_test_raw = undersampling(x, y, test_size, rs=i,sm=False)

                x_train = x_train_raw.to_numpy()
                x_test = x_test_raw.to_numpy()
                y_train = y_train_raw.to_numpy()
                y_test = y_test_raw.to_numpy()

        
                
                # semi supervisioned models
                # y_pred,y_pred_proba,model_cv = RSLocalOutlierFactor(x_train, y_train, x_test, y_test,n_jobs=N_JOBS)
                # sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
                # results['scaling_method'].append(scaling)
                # results['iteration'].append(i)
                # results['fold'].append(j)
                # results['oversampling_technique'].append(oversampling_technique)
                # results['model_name'].append('LOF')                        
                # results['F1'].append(f1)
                # results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                # results['SEN'].append(sen)
                # results['SPE'].append(spe)
                # log.info(f'LOF.....................: {f1}')

                
                # y_pred,y_pred_proba,model_cv = RSOneClassSVM(x_train, y_train, x_test, y_test,n_jobs=N_JOBS)
                # sen, spe, f1, roc, jac, fmi, mcc,acc = computer_scores(y_test, y_pred)
                # results['scaling_method'].append(scaling)
                # results['iteration'].append(i)
                # results['fold'].append(j)
                # results['oversampling_technique'].append(oversampling_technique)
                # results['model_name'].append('OneClassSVM')
                # results['ACC'].append(acc)
                # results['F1'].append(f1)
                # results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                # results['SEN'].append(sen)
                # results['SPE'].append(spe)
                # log.info(f'OneClassSVM.............: {f1}')


                # y_pred,y_pred_proba,model_cv = RSIsolationForest(x_train, y_train, x_test, y_test,n_jobs=N_JOBS)
                # sen, spe, f1, roc, jac, fmi, mcc,acc = computer_scores(y_test, y_pred)
                # results['scaling_method'].append(scaling)
                # results['iteration'].append(i)
                # results['fold'].append(j)
                # results['oversampling_technique'].append(oversampling_technique)
                # results['model_name'].append('RSIsolationForest')
                # results['ACC'].append(acc)
                # results['F1'].append(f1)
                # results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                # results['SEN'].append(sen)
                # results['SPE'].append(spe)
                # log.info(f'RSIsolationForest.......: {f1}')


                # supervisioned models
                y_pred,y_pred_proba,model_cv = RSRF(x_train, y_train, x_test, y_test,n_jobs=N_JOBS,random_state=i)
                sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, auc,acc_class_1,acc_class_2 = computer_scores_roc(y_test, y_pred,y_pred_proba)
                results['scaling_method'].append(scaling)
                results['iteration'].append(i)
                # results['fold'].append(j)
                results['oversampling_technique'].append(oversampling_technique)
                results['model_name'].append('RF')
                # results['ACC'].append(acc)
                results['F1'].append(f1)
                results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                results['acc-class-1'].append(acc_class_1)
                results['acc-class-2'].append(acc_class_2)
                results['SEN'].append(sen)
                results['SPE'].append(spe)
                results['TPR'].append(tpr)
                results['FPR'].append(fpr)
                results['AUC'].append(auc)
                results
                log.info(f'RandomForestClassifier..: {f1}')

                y_pred,y_pred_proba,model_cv = RSNN(x_train, y_train, x_test, y_test,n_jobs=N_JOBS,random_state=i)
                sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, auc, acc_class_1,acc_class_2= computer_scores_roc(y_test, y_pred,y_pred_proba)
                results['scaling_method'].append(scaling)
                results['iteration'].append(i)
                # results['fold'].append(j)
                results['oversampling_technique'].append(oversampling_technique)
                results['model_name'].append('KNN')
                # results['ACC'].append(acc)
                results['F1'].append(f1)
                results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                results['SEN'].append(sen)
                results['SPE'].append(spe)
                results['TPR'].append(tpr)
                results['FPR'].append(fpr)
                results['AUC'].append(auc)
                results['acc-class-1'].append(acc_class_1)
                results['acc-class-2'].append(acc_class_2)
                log.info(f'KNN ....................: {f1}')


                y_pred,y_pred_proba,model_cv = RSDT(x_train, y_train, x_test, y_test,random_state=i)
                sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, auc,acc_class_1,acc_class_2 = computer_scores_roc(y_test, y_pred,y_pred_proba)
                results['scaling_method'].append(scaling)
                results['iteration'].append(i)
                # results['fold'].append(j)
                results['oversampling_technique'].append(oversampling_technique)
                results['model_name'].append('DT')
                # results['ACC'].append(acc)
                results['F1'].append(f1)
                results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                results['SEN'].append(sen)
                results['SPE'].append(spe)
                results['TPR'].append(tpr)
                results['FPR'].append(fpr)
                results['AUC'].append(auc)
                results['acc-class-1'].append(acc_class_1)
                results['acc-class-2'].append(acc_class_2)
                log.info(f'DT .....................: {f1}')
                

                y_pred,y_pred_proba,model_cv = RSXGB(x_train, y_train, x_test, y_test,random_state=i, synclass=False)
                sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, auc,acc_class_1,acc_class_2= computer_scores_roc(y_test, y_pred,y_pred_proba)
                results['scaling_method'].append(scaling)
                results['iteration'].append(i)
                # results['fold'].append(j)
                results['oversampling_technique'].append(oversampling_technique)
                results['model_name'].append('XGB')
                # results['ACC'].append(acc)
                results['F1'].append(f1)
                results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                results['SEN'].append(sen)
                results['SPE'].append(spe)
                results['TPR'].append(tpr)
                results['FPR'].append(fpr)
                results['AUC'].append(auc)
                results['acc-class-1'].append(acc_class_1)
                results['acc-class-2'].append(acc_class_2)
                log.info(f'XGBoost .....................: {f1}')

                y_pred,y_pred_proba,model_cv = RS_stacking_classifier(x_train, y_train, x_test, y_test)
                sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, auc,acc_class_1,acc_class_2= computer_scores_roc(y_test, y_pred,y_pred_proba)
                results['scaling_method'].append(scaling)
                results['iteration'].append(i)
                # results['fold'].append(j)
                results['oversampling_technique'].append(oversampling_technique)
                results['model_name'].append('Stacking')
                # results['ACC'].append(acc)
                results['F1'].append(f1)
                results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                results['SEN'].append(sen)
                results['SPE'].append(spe)
                results['TPR'].append(tpr)
                results['FPR'].append(fpr)
                results['AUC'].append(auc)
                results['acc-class-1'].append(acc_class_1)
                results['acc-class-2'].append(acc_class_2)
                log.info(f'Stacking .....................: {f1}')

                # y_pred,y_pred_proba,model_cv = RSSVM(x_train, y_train, x_test, y_test)
                # sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
                # results['scaling_method'].append(scaling)
                # results['iteration'].append(i)
                #results['fold'].append(j)
                # results['oversampling_technique'].append(oversampling_technique)
                # results['model_name'].append('SVM')
                # results['F1'].append(f1)
                # results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                # results['SEN'].append(sen)
                # results['SPE'].append(spe)
                # log.info(f'SVM ....................: {f1}')

                # y_pred,y_pred_proba,model_cv = SVM_class_weight(x_train, y_train, x_test, y_test,n_jobs=N_JOBS)
                # sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
                # results['scaling_method'].append(scaling)
                # results['iteration'].append(i)
                # results['fold'].append(j)
                # results['oversampling_technique'].append(oversampling_technique)
                # results['model_name'].append('SVM-weight')
                # results['F1'].append(f1)
                # results['ROC'].append(roc)
                # results['IOU'].append(jac)
                # results['FMI'].append(fmi)
                # results['MCC'].append(mcc)
                # results['SEN'].append(sen)
                # results['SPE'].append(spe)
                # log.info(f'SVM Weight..............: {f1}')
                
                
                seconds_j = (time.time()- ini_j)
                minutes_j = round(seconds_j / 60, 2)
                log.info(f'iteration {i} oversampling {oversampling_technique} finished in {minutes_j} minutes')

    df = pd.DataFrame(results)
    filename = f'{experiment_info["dataset_info"]["dsetname"]}_{experiment_info["experiment_name"]}'

    # Calculate mean values for each metric based on scaling method, oversampling technique, and model
    df_describe = df.groupby(['scaling_method', 'oversampling_technique', 'model_name']).mean().reset_index()
    df_describe.drop(['iteration'],axis=1,inplace=True)
    df_describe = df_describe.round(2)
    df_describe = get_table_with_std(df,df_describe)
    df_describe.to_csv(f'out/{folder}/Describe_{experiment_info["dataset_info"]["dsetname"]}_experiment_{test_size}.csv',index=False)
    df.to_csv(f'out/{folder}/Raw_Results/{experiment_info["dataset_info"]["dsetname"]}_{test_size}.csv', index=False)

    # show sample
    log.info('-' * 30)
    log.info(df.describe())
    log.info('-' * 30)



if __name__ == '__main__':
    ini = time.time()
    experiments = [
        {
            "experiment_name": "Pipeline_Hemophilia_A-FVIII",
            "dataset_info": {
                "dsetname" : "hemophilia-A-FVIII",
                "dataset_path": "./hemo_synclass/",
                "dataset_sep": ",",
            },
            "pre_processing_func":pre_processing_hemoA
        },

        {
            "experiment_name": "Pipeline_Hem_RIN2r7e",
            "dataset_info":{
                "dsetname": "RIN-2R7E-label",
                "dataset_path": "./hemo_synclass/",
                "dataset_sep":',',
                
            },
            "pre_processing_func": pre_processing_rin
        },
        
        {
            "experiment_name": "Pipeline_Hemo_B",
            "dataset_info":{
                "dsetname": "HemB_Dataset_SENAI_v5a",
                "dataset_path": "./hemo_synclass/",
                "dataset_sep":'\t',
                
            },
            "pre_processing_func":pre_processing_hemoB
        }
    ]

    for experiment_info in experiments:
      
        log.info('-' * 30)
        log.info(f'Experiment {experiment_info["experiment_name"]}')
        log.info('-' * 30)
        percentage = [0.1,0.15,0.2,0.25]
        for test_size in [0.1,0.15,0.2,0.25]:
            run_all_pipeline_under_smote_techiques(experiment_info,folder=f'pipeline_undersampling/{experiment_info["experiment_name"]}', test_size = test_size)


    seconds = (time.time()- ini)
    minutes = seconds / 60
    log.info(f'Run Classification pipeline in {seconds:.3f} seconds or {minutes:.3f} minutes.')