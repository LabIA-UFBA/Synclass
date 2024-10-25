import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE, ADASYN,BorderlineSMOTE
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def label_encoder(array):
    return LabelEncoder().fit_transform(array)


def standard_scaler(x: pd.DataFrame):
    '''
    Standardize features by removing the mean and scaling to unit variance.
    The standard score of a sample x is calculated as:
    z = (x - u) / s
    '''
    scaler = StandardScaler().fit(x)
    
    return pd.DataFrame(scaler.transform(x), columns = x.columns)


def undersampling_random(x_features, 
                         Y, 
                         percentage, 
                         rs, 
                         at='target'):
    X = x_features.copy()
    X[at] = Y
    # surffle
    X = shuffle(X, random_state=rs)
    # count class proportions
    proportions = Counter(X[at])
    # find minor class
    class_minority = min(proportions, key=proportions.get)
    size_minority  = proportions[class_minority]
    # get train and test size
    p_test = np.ceil(size_minority * percentage).astype('int')
    p_train = (size_minority - p_test)
    # select partitions
    train, test = [], []
    for classe in X[at].unique():
        df_class = X[X[at] == classe]
        test.append(df_class.iloc[:p_test])
        train.append(df_class.iloc[p_test:p_train])
    # define new train test and split
    df_train = pd.concat(train)
    df_test = pd.concat(test)
    y_train = df_train[at]
    y_test = df_test[at]
    x_train = df_train.drop([at], axis=1)
    x_test = df_test.drop([at], axis=1)   
    
    return x_train, y_train, x_test, y_test


def over_sampling_smote(x_train, y_train, k_neighbors=5):
    '''
    Synthetic Minority Oversampling Technique (SMOTE)
    '''
    X_resampled, y_resampled = SMOTE(k_neighbors=k_neighbors).fit_resample(x_train, y_train)

    return X_resampled, y_resampled


def over_sampling_smote_tek(x_train,y_train):

    X_resampled,y_resampled = SMOTETomek().fit_resample(x_train,y_train)

    return X_resampled,y_resampled

def over_sampling_borderline_smote(x_train,y_train):
    X_resampled,y_resampled = BorderlineSMOTE().fit_resample(x_train,y_train)

    return X_resampled,y_resampled

def over_sampling_adasyn(x_train, y_train):
    '''
    Adaptive Synthetic (ADASYN)
    '''
    X_resampled, y_resampled = ADASYN().fit_resample(x_train, y_train)

    return X_resampled, y_resampled


def over_sampling_smote_enn(x_train, y_train):
    '''
    Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.
    '''
    X_resampled, y_resampled = SMOTEENN().fit_resample(x_train, y_train)

    return X_resampled, y_resampled


def random_under_sampler(x, y, rs=0):
    '''
    Under-sample the majority class(es) by randomly 
    picking samples with or without replacement.
    '''
    rus = RandomUnderSampler(random_state=rs)
    X_resampled, y_resampled = rus.fit_resample(x, y)

    return X_resampled, y_resampled


def random_under_near_miss(x, y, rs=0):
    '''
     Knn approach to unbalanced data distributions: 
     a case study involving information extraction.
    '''
    nm1 = NearMiss(version=1)
    X_resampled_nm1, y_resampled = nm1.fit_resample(x, y)

    return X_resampled_nm1, y_resampled


def pre_processing(data: pd.DataFrame, 
                   stand=False, 
                   remove_nan=False, 
                   undersampling=None, 
                   oversampling=None):
    '''
     main to pre processing
    '''
    data_norm = data.copy()
    if stand:
        data_norm = standard_scaler(data)
    if remove_nan:
        data_norm = data_norm.dropna()

    return data_norm