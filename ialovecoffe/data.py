# io, system, data
import os
import time
import pickle
import pandas as pd
import numpy as np
from math import sqrt
from loguru import logger as log
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime
from collections import Counter
# preprocessing
from ialovecoffe.cvstrat import *
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# models
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

from sklearn.utils import shuffle
from ialovecoffe.pre_processing import *

simplefilter("ignore", category=ConvergenceWarning)


def load_model(data_path="data/data_hemophilia.pkl"):
    df = df = pickle.load(open(data_path, "rb"))
    
    return  df

def load_data(dataset_info):
    df = pd.read_csv(dataset_info.dataset_path,sep=dataset_info.sep)
    
    if dataset_info.dropna:
        df = df.dropna()

    x,y = dataset_info.encoding_x(df.drop(dataset_info.drops,axis=1)), dataset_info.encoding_y(df[dataset_info.target])
    
    return x,y,df



def feature_selection(algorithm, k_best, data_train_X, data_train_y):

    if(algorithm == 'rfe'):
        regre = LogisticRegression(solver='lbfgs')
        model = RFE(regre, n_features_to_select=k_best, step=1)
        return model.fit_transform(data_train_X, data_train_y)

    if(algorithm == 'kbest'):
        kbest = SelectKBest(score_func=f_classif, k=k_best)
        return kbest.fit_transform(data_train_X, data_train_y)

    if(algorithm == 'extra_tree'):
        clf = ExtraTreesClassifier(n_estimators=100).fit(data_train_X, data_train_y)
        model = SelectFromModel(clf, prefit=True)
        return model.transform(data_train_X)

    if(algorithm == 'linear_svc'):
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data_train_X, data_train_y)
        model = SelectFromModel(lsvc, prefit=True)  
        return model.transform(data_train_X)
    
    return data_train_X


# Revisar funçao 
def pre_processing(df, target_feature,scaling ="NoScaling",
                       algorithm=None, 
                       k_best='all', 
                       atts=None):

    scaling_techiniques = {
        "NoScaling": lambda data_train_x: data_train_X,
        "MinMax" : MinMaxScaler().fit_transform,
        "Standard" : StandardScaler().fit_transform
    }

    data_train_y = df[target_feature]
    data_train_y = data_train_y.replace(['No', 'Yes'], [1, -1]).to_numpy()
    
    if atts:
        data_train_X = df[atts].to_numpy() # set attributes
        data_train_X = data_train_X.reshape(data_train_X.shape[0], len(atts))
    else: 
        data_train_X = df.drop([target_feature], axis=1).to_numpy()
    

    if algorithm:
        data_train_X = feature_selection(algorithm, k_best, data_train_X, data_train_y)
    
    data_train_X = scaling_techiniques[scaling](data_train_X)
    
    return data_train_X, data_train_y, df


def apply_smote(x, y):
    x_smt, y_smt = SMOTE().fit_resample(x, y)
    
    return  x_smt, y_smt


def data_splitting(x, y, df, 
                   n_folds = 3, 
                   smote=False, 
                   test_size=5):

    data_folds = []

    for fold_id in range(n_folds):
        
        index_yes, index_no = create_cv_balanced(df['inhibit'], test_size=test_size)

        list_index = np.concatenate((index_yes[fold_id], index_no[fold_id]))

        # test
        x_test = x[list_index,].copy()
        y_test = y[list_index,].copy()
        
        # train
        x_train = np.delete(x, list_index, axis=0)
        y_train = np.delete(y, list_index)
            
        if smote:
            x_train, y_train = apply_smote(x_train, y_train)

        data_folds.append([x_train, y_train, x_test, y_test])


    return data_folds


def load_data_fold(iteration, fold):
        
        folder_name = f'folds/iter-{iteration}-fold-{fold}/'
        x_train = np.load(folder_name+'x_train.npy')
        y_train = np.load(folder_name+'y_train.npy')
        x_test = np.load(folder_name+'x_test.npy')
        y_test = np.load(folder_name+'y_test.npy')

        return x_train, y_train, x_test, y_test


def save_data_fold(iteration, fold, x_train, y_train, x_test, y_test, folder='folds/'):
    try:
        folder_name = f'{folder}iter-{iteration}-fold-{fold}/'
            
        # create folder with metaname
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # salve x and y
        np.save(folder_name+'x_train.npy', x_train)
        np.save(folder_name+'y_train.npy', y_train)
        np.save(folder_name+'x_test.npy', x_test)
        np.save(folder_name+'y_test.npy', y_test)

        return True
    except:
        return False


def under_sampling_equals_class(ydata, size_upper=None):
    
    populations = Counter(ydata)
    target_lower, target_upper  = min(populations), max(populations)
    len_lower = populations[target_lower]
    len_upper = populations[target_upper]
    
    # index lower
    idx_lower = np.where(ydata==target_lower)[0]
    idx_upper = np.where(ydata==target_upper)[0]
    
    # random indexs
    rnd_indexs = np.random.choice(idx_upper, size=len(idx_lower))
    
    if size_upper == 'all':
        rnd_indexs = idx_upper
    
    if isinstance(size_upper, int):
        rnd_indexs = np.random.choice(idx_upper, size=size_upper)
    
    
    return idx_lower.tolist(), list(rnd_indexs)

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


def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2

	return sqrt(distance)


def kmeans_split_data(k_clusters = 20, num_neighbors = 3, data_frame = any):

    km = KMeans(n_clusters=k_clusters).fit(data_frame)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data_frame.index.values
    cluster_map['cluster'] = km.labels_

    new_df = pd.DataFrame()

    for i in range(0, k_clusters):
        neighbors = {}    

        list_index = cluster_map[cluster_map.cluster == i]['data_index']
        for index in list_index:
            neighbors.update({index: euclidean_distance(km.cluster_centers_[i], data_frame.loc[index])})
        a = sorted(neighbors.items(), key=lambda x: x[1])    
        
        for j in a[0:num_neighbors]:
            new_df = new_df.append(data_frame.loc[j[0]])

    return new_df


def read_A_thrombosis_non_thrombosis_v5(path_src = 'data/inputSets_for_thrombosis_pipeline/'):
    df = pd.read_csv(f'{path_src}thrombosis_non_thrombosis_v5.csv', sep='\t')

    Y = df['type'].copy()
    df.drop(['node', 'type'], inplace=True, axis=1)
    X = df
    return X, Y, df