import sys
import numpy as np
import pandas as pd
import os
import random
from loguru import logger as log
from ialovecoffe.models import RSDT_synclass,RSRF_synclass,RSKNN_synclass,RSXGB,RS_stacking_classifier_synclass
from ialovecoffe.data import load_data
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from synclass.pipeline_synclass import run_experiment,get_best_thresh_metrics,get_table_with_std
from tqdm import tqdm


def run_synclass_pipeline(test_percentage, experiment_info, NUM_ITER = 5,folder='synclass'):
    os.makedirs(f'out/{folder}/Raw_Results',exist_ok=True)
    #dsetname = "hemophilia-A-FVIII"0
    # data = pd.read_csv(f'./hemo_synclass/{dsetname}.csv',delimiter= ',')
    # data = pd.read_csv(experiment_data.dataset.path,delimiter=experiment_data.dataset.delimiter)
    # data = data.dropna()

    #x,y = data.drop(['group'],axis=1), LabelEncoder().fit_transform(data["group"])

    #df = pd.read_csv('./hemo_synclass/hemophilia-A-FVIII.csv')
    
    # y = df['group'].replace({'Others': 0, 'Severe': 1})
    # x = df.drop(['group'], axis=1)
    full_path = experiment_info["dataset_info"]["dataset_path"] +experiment_info["dataset_info"]["dsetname"] + '.csv'

    dataset = pd.read_csv(full_path,sep = experiment_info["dataset_info"]["dataset_sep"])

    x,y,dataset = experiment_info["pre_processing_func"](dataset)
    
    # run
  
    #learners = experiment_data.learners
    learners = [
        ("Decision Tree",RSDT_synclass),
        ("Random Forest",RSRF_synclass),
        ("KNN",RSKNN_synclass),
        ("XGBoost",RSXGB),
        ("Stacking",RS_stacking_classifier_synclass)
    ]


    df_folds = run_experiment(x, y, NUM_ITER,test_percentage, experiment_info["dataset_info"]["dsetname"],learners=learners,smote=False)
    df_describe = get_best_thresh_metrics(df_folds,"F1")

    filename = f'{experiment_info["dataset_info"]["dsetname"]}_{experiment_info["experiment_name"]}'

    df_describe = get_table_with_std(df_folds,df_describe)
    # df_describe.to_csv(f'out/{folder}/Describe_{experiment_info["experiment_name"]}_experiment_{test_percentage}.csv', index=False)
    
    # df_folds.to_csv(f'out/{folder}/Raw_Results/{experiment_info["experiment_name"]}_experiment_{test_percentage}.csv', index=False)
    
    

def pre_processing_rin(df):
    df = df.dropna()
    y = df['Label.Activity'].replace({'low': 0, 'high': 1})
    df["res"] = LabelEncoder().fit_transform(df.loc[:,"res"])

    x = df.drop(['node_', 'node', 'Activity', 'Label.Activity'],axis= 1)

    return x,y,df


def pre_processing_hemoB(df):
    label_encoder = LabelEncoder()
    df['Domain'] = label_encoder.fit_transform(df.loc[:,'Domain'])
    df['Protein_Change'] = label_encoder.fit_transform(df.loc[:,'Protein_Change'])
    df['aa1'] = label_encoder.fit_transform(df.loc[:,'aa1'])
    df['aa2'] = label_encoder.fit_transform(df.loc[:,'aa2'])
    df.fillna(0, inplace=True)
    
    y = df['Reported_Severity'].replace({'Others': 0, 'Severe': 1})
    x = df.drop(['Reported_Severity'], axis=1)
    
    return x, y, df


def pre_processing_hemoA(df):
    y = df['group'].replace({'Others': 0, 'Severe': 1})
    x = df.drop(['group'], axis=1)

    return x,y,df

def pre_processing_thrombosis(df):
    
    df['type_bin'] = df['type'].replace({'Non_thrombosis': 0, 'Thrombosis': 1})
    y = df['type_bin'].copy()
    df.drop(['node', 'type', 'type_bin'], inplace=True, axis=1)
    x = df
    return x, y, df

def pre_processing_loan(df):
    le = LabelEncoder()
    df[df.columns[1]] = le.fit_transform(df[df.columns[1]])

    # x = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values
    x,y = df.drop("not.fully.paid",axis=1), df["not.fully.paid"]

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    return x, y, df

if __name__ == '__main__':
    ITERS = 1
    random.seed(10)

    experiments = [
        {
            "experiment_name": "Synclass_Hemophilia_A-FVIII",
            "dataset_info": {
                "dsetname" : "hemophilia-A-FVIII",
                "dataset_path": "data/datasets/A-FVIII/",
                "dataset_sep": ",",
            },
            "pre_processing_func":pre_processing_hemoA
        },

        {
            "experiment_name": "Synclass_Hem_RIN2r7e",
            "dataset_info":{
                "dsetname": "RIN-2R7E-label",
                "dataset_path": "data/datasets/FV-VIII-RIN/",
                "dataset_sep":',',
                
            },
            "pre_processing_func": pre_processing_rin
        },
        
        {
            "experiment_name": "Synclass_Hemo_B",
            "dataset_info":{
                "dataset_path": "data/datasets/hemophilia-b/",
                "dataset_sep":'\t',
                
            },
            "pre_processing_func":pre_processing_hemoB
        }
    ]

    for experiment_info in experiments:
        log.info('-' * 30)
        log.info(f'Experiment {experiment_info["experiment_name"]}')
        log.info('-' * 30)

        for test_size in [0.1]:#,0.15,0.2,0.25]:

        
            run_synclass_pipeline(test_size, experiment_info,NUM_ITER=ITERS, folder=f'synclass/{experiment_info["experiment_name"]}')
            