import pandas as pd
import numpy as np
import random

def create_cv(data_labels, test_size = 5, reference='Yes'):
    index = (np.where(data_labels == reference)[0]).tolist()    
    folds = len(index)//test_size
    cv_indexes = []

    for i in np.arange(folds-1):
        temp = random.sample(index, test_size)  
        cv_indexes.append(temp)
        index = [x for x in index if x not in temp]
    
    cv_indexes.append(index)

    return cv_indexes

def create_cv_balanced(data_labels, test_size = 5, reference='Yes'):    
    index = (np.where(data_labels == reference)[0]).tolist()
    index_out = (np.where(data_labels != reference)[0]).tolist()  
    index_out = random.sample(index_out, 15)   
    folds = len(index)//test_size
    cv_indexes = []
    cv_indexes_out = []

    for i in np.arange(folds-1):
        temp = random.sample(index, test_size)
        temp_out = random.sample(index_out, test_size)   
        cv_indexes.append(temp)
        cv_indexes_out.append(temp_out)
        index = [x for x in index if x not in temp]
        index_out = [x for x in index_out if x not in temp_out]

    cv_indexes.append(index)
    cv_indexes_out.append(index_out)

    return cv_indexes, cv_indexes_out