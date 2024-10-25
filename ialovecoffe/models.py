import pickle
import numpy as np
import xgboost as xgb
from loguru import logger as log
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
# tuning
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials



def RSOneClassSVM(X_train, y_train, X_test, y_test, measure='f1_macro',n_jobs = -1):
    
    pipe = Pipeline([
        ("clf", OneClassSVM(max_iter = 500))
    ])

    params = {
        "clf__kernel": ['rbf', 'poly', 'sigmoid'],
        "clf__gamma": np.arange(0.1, 5, 0.3),
        "clf__coef0": np.arange(0, 5, 0.2),
        "clf__degree": np.arange(1, 8, 1)
    }

    model_cv = RandomizedSearchCV(pipe, params, n_iter=50, scoring=measure, n_jobs = n_jobs, random_state=1)

    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_test)
    #y_pred_prob = model_cv.predict_proba(X_test)
    
    y_pred = np.where(y_pred < 0, 1, 0)
    
    y_pred_prob = model_cv.decision_function(X_test)

    #print(y_pred)
    return y_pred, y_pred_prob, model_cv


def RSKNN_synclass(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro',random_state=1):
    params = {'n_neighbors':np.arange(1, 25, 1), 
              'weights':np.array(['uniform', 'distance']),
              'p':np.arange(1, 4, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
    model = KNeighborsClassifier()
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring='f1_macro', n_jobs = -1, n_iter=50,random_state=random_state)

    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)

    model_cv.fit(X_train, y_train)

    y_pred_prob = model_cv.predict_proba(X_test)
    y_pred = model_cv.predict(X_test)
    
    return y_pred, y_pred_prob, model_cv


def RSLocalOutlierFactor_ml(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro'):
    
    pipe = Pipeline([
        ("clf", LocalOutlierFactor(novelty=True))
    ])

    params = {'clf__algorithm':['ball_tree', 'kd_tree', 'brute'], 
              'clf__n_neighbors':[5, 10, 15, 20, 25, 30, 35, 40, 45], 
              'clf__contamination': [0.35, 0.40, 0.45, 0.5], 
              'clf__leaf_size': [5, 10, 20, 30, 40],
              'clf__metric': ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'], 
              'clf__p':[1,2,3,4,5]}

    model_cv = RandomizedSearchCV(pipe, params, n_iter=50, 
                                  scoring=metric, 
                                  n_jobs = -1, 
                                  random_state=1)

    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_test)
    #y_pred_prob = model_cv.predict_proba(X_test)

    y_pred = np.where(y_pred < 0, 1, 0)
    y_pred_prob = model_cv.decision_function(X_test)
    
    return y_pred, y_pred_prob, model_cv


def RSLocalOutlierFactor(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro',n_jobs = -1):
    
    pipe = Pipeline([
        ("clf", LocalOutlierFactor(novelty=True))
    ])

    params = {'clf__algorithm':['ball_tree', 'kd_tree', 'brute'], 
              'clf__n_neighbors':[5, 10, 15, 20, 25, 30, 35, 40, 45], 
              'clf__contamination': [0.35, 0.40, 0.45, 0.5], 
              'clf__leaf_size': [5, 10, 20, 30, 40],
              'clf__metric': ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'], 
              'clf__p':[1,2,3,4,5]}

    model_cv = RandomizedSearchCV(pipe, params, n_iter=50, 
                                  scoring=metric, 
                                  n_jobs = n_jobs, 
                                  random_state=1)

    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)

    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_test)
    #y_pred_prob = model_cv.predict_proba(X_test)

    y_pred = np.where(y_pred < 0, 1, 0)
    y_pred_prob = model_cv.decision_function(X_test)
    
    return y_pred, y_pred_prob, model_cv


def hybrid_distance(vectA, vectB):
    list_index = np.array([0, 1, 2, 8, 14])
    
    vectA_hamming = vectA[list_index]
    vectB_hamming = vectB[list_index]

    vectA_cosine = np.delete(vectA, list_index)
    vectB_cosine = np.delete(vectB, list_index)

    cosine_distance = distance.cosine(vectA_cosine, vectB_cosine)
    hamming_distance = distance.hamming(vectA_hamming, vectB_hamming)

    return cosine_distance + hamming_distance


def RSLocalOutlierFactorTunedStatic(X_train, y_train, X_test, y_test):
    
    #tuner = LOF_AutoTuner(data = X_train, k_max =  50, c_max = 0.4)
    
    #run tuner
    #params = tuner.run()
    #print(type(params))
    #model_cv = LocalOutlierFactor(novelty=True, p=2, n_neighbors=5, metric = 'cosine', leaf_size=20, contamination=0.45, algorithm='brute')
    model_cv = LocalOutlierFactor(novelty=True, p=2, n_neighbors=5, metric = 'cosine', leaf_size=5, contamination=0.35, algorithm='brute')
    #model_cv = LocalOutlierFactor(novelty=True, p=4, n_neighbors=10, metric = 'cosine', leaf_size=10, contamination=0.5, algorithm='brute')
    model_cv.fit(X_train)
    y_pred = model_cv.predict(X_test)
    
    return model_cv, y_pred


def RSIsolationForest(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro',n_jobs = -1):

    pipe = Pipeline([
    ("clf", IsolationForest())
    ])

    params = {'clf__n_estimators': np.array(list(range(10, 100, 5))),
            'clf__contamination': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            #'clf__max_features': np.arange(start=2, stop=(X_train.shape[1]+1)),
            'clf__bootstrap': np.array([True, False])}

    model_cv = RandomizedSearchCV(pipe, params, n_iter=50, scoring='f1_macro', n_jobs = n_jobs, random_state=1)

    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_test)
    #y_pred_prob = model_cv.predict_proba(X_test)

    y_pred = np.where(y_pred < 0, 1, 0)
    y_pred_prob = model_cv.decision_function(X_test)
    
    return y_pred, y_pred_prob, model_cv


def RSEllipticEnvelope(X_train, y_train, X_test, y_test):
    pipe = Pipeline([
    ("clf", IsolationForest())
    ])

    params = {}# here

    model_cv = RandomizedSearchCV(pipe, params, n_iter=1000, scoring='f1_macro', n_jobs = -1, random_state=1)

    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_test)
    
    return model_cv, y_pred


def RSAC(X_train, y_train, X_test, y_test):

    pipe = Pipeline([
    ("clf", AgglomerativeClustering())
    ])

    params = {'clf__eps':np.arange(0.5, 15, 0.5), 
              'clf__min_samples': np.arange(3,20,1)}

    model_cv = RandomizedSearchCV(pipe, params, n_iter=1000, scoring='f1_macro', n_jobs = -1, random_state=1)

    model_cv.fit(X_train, y_train)
    
    cluster = DBSCAN()
    cluster.set_params = model_cv.best_params_
    cluster.fit(X_train)
    y_pred = cluster.fit_predict(X_test)
    
    return cluster, y_pred


def RSDBScan(X_train, y_train, X_test, y_test):

    pipe = Pipeline([
    ("clf", DBSCAN())
    ])

    params = {'clf__eps':np.arange(0.5, 15, 0.5), 
              'clf__min_samples': np.arange(3,20,1)}

    model_cv = RandomizedSearchCV(pipe, params, n_iter=1000, scoring='f1_macro', n_jobs = -1, random_state=1)

    model_cv.fit(X_train, y_train)
    
    cluster = DBSCAN()
    cluster.set_params = model_cv.best_params_
    cluster.fit(X_train)
    y_pred = cluster.fit_predict(X_test)
    
    return cluster, y_pred


def save_model(model, filename, folder='saved_models/'):
    # Save to file in the current working directory
    with open(folder + filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(filename, folder='saved_models/'):    
    # Load from file
    model = None
    with open(folder + filename, 'rb') as file:
        model = pickle.load(file)
    return model

'''
Runing classification models
'''
def RSXgboost(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro',threshold = 0.5,random_state = 1):
    params = {
            "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],
            "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight" : [ 1, 3, 5, 7 ],
            "gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }
    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=random_state)
    classifier = xgb.XGBClassifier(verbosity = 0, silent=True,seed=random_state)
    best_model = RandomizedSearchCV(classifier, param_distributions=params,
                                   n_iter=50, n_jobs = -1, 
                                   scoring=metric, cv=cv, 
                                   verbose=0,random_state= random_state)
    
    best_model.fit(X_train, y_train, verbose=False)
    y_pred_prob = best_model.predict_proba(X_test)
    y_pred_m = best_model.predict(X_test)
    y_pred = (y_pred_prob[:,1] >= threshold).astype(int)
    print("model")
    print(y_pred_m)
    print(y_pred)
    return y_pred, y_pred_prob, best_model


def RSXGB(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro',random_state=1,synclass = True):
    params = {'max_depth':np.arange(1, 25, 1), 
              'eta':np.arange(0.1, 0.5, 0.01),
              'lambda':np.arange(0, 1, 0.1)
             }
    
   
    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=random_state)
    model = xgb.XGBClassifier(objective="binary:logistic", verbosity = 0, silent=True, random_state=random_state)
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring=metric, n_jobs = -1, n_iter=50,random_state=random_state)
    
    if synclass:
        X_train = X_train.values.reshape(-1,1)
        X_test= X_test.values.reshape(-1,1)

    model_cv.fit(X_train, y_train)
    

    y_pred_proba = model_cv.predict_proba(X_test)

    y_pred = model_cv.predict(X_test)
    
    return y_pred, y_pred_proba,model_cv


def hyperopt_train_test(params):
    x = params['data']['x']
    y = params['data']['y']
    clf = xgb.XGBClassifier(objective="binary:logistic", verbosity = 0, 
                            silent=True, random_state=42, **params['params'])

    return cross_val_score(clf, x, y, cv=10, scoring='f1_macro').mean()


def _hyperopt_train_test(params):
    x = params['data']['x']
    y = params['data']['y']
    model = params['model'].set_params(**params['params'])
    
    return cross_val_score(model, x, y, cv=10, scoring='f1_macro').mean()



def f(params):
    mean_score = hyperopt_train_test(params)

    return {'loss': -mean_score, 'status': STATUS_OK}


def RSXGB_hiperopt(X_train, y_train, X_test, y_test, prob=False):
    '''
    Parameter Tuning with Hyperopt (Bayesian optimization)
    '''
    space = {'data': {'x': X_train, 'y': y_train},
            'params': {'max_depth': hp.choice('max_depth', np.arange(1, 25, 1)),
                        'eta': hp.choice('eta', np.arange(0.1, 0.5, 0.01)),
                        'lambda': hp.choice('lambda', np.arange(0, 1, 0.1))}
            }
    trials = Trials()
    
    best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)

    best_model = xgb.XGBClassifier(objective="binary:logistic", verbosity = 0, silent=True, random_state=42, **best)

    best_model.fit(X_train, y_train)
    
    if prob:
        y_pred = best_model.predict_proba(X_test)
    else:
        y_pred = best_model.predict(X_test)
    
    return best_model, y_pred


def RSRF(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro',n_jobs = -1,random_state=1):
    params = {'n_estimators':np.arange(1, 25, 1), 
              'max_depth':np.arange(1, 25, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=random_state)
    model = RandomForestClassifier(random_state=random_state)
    model_cv = RandomizedSearchCV(model, params, cv = cv, 
                                  scoring=metric, 
                                  n_jobs = n_jobs, 
                                  n_iter=50,random_state=random_state)

    model_cv.fit(X_train, y_train)
    
    
    y_pred_prob = model_cv.predict_proba(X_test)
    y_pred = model_cv.predict(X_test)
    
    return y_pred, y_pred_prob, model_cv


def RSRF_synclass(X_train, y_train, X_test, y_test, prob=False,metric='f1_macro',random_state=1):
    params = {'n_estimators':np.arange(1, 25, 1), 
              'max_depth':np.arange(1, 25, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
    model = RandomForestClassifier(random_state=random_state)
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring=metric, n_jobs = -1, n_iter=50,random_state=random_state)
    
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)

    model_cv.fit(X_train, y_train)

    y_pred_prob = model_cv.predict_proba(X_test)
    y_pred = model_cv.predict(X_test)
    
    return y_pred, y_pred_prob, model_cv


def RSNN(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro',n_jobs = -1,random_state=1):
    params = {'n_neighbors':np.arange(1, 25, 1), 
              'weights':np.array(['uniform', 'distance']),
              'p':np.arange(1, 4, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
    model = KNeighborsClassifier()
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring='f1_macro', n_jobs = n_jobs, n_iter=50,random_state=random_state)

    model_cv.fit(X_train, y_train)
    
   
    y_pred_prob = model_cv.predict_proba(X_test)
    y_pred = model_cv.predict(X_test)
    
    return y_pred, y_pred_prob, model_cv

def RSNN_synclass(X_train, y_train, X_test, y_test, prob=False,random_state=1):
    params = {'n_neighbors':np.arange(1, 25, 1), 
              'weights':np.array(['uniform', 'distance']),
              'p':np.arange(1, 4, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
    model = KNeighborsClassifier(random_state=random_state)
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring='f1_macro', n_jobs = -1, n_iter=50,random_state=random_state)

    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)

    model_cv.fit(X_train, y_train)   
    y_pred_prob = model_cv.predict_proba(X_test)
    y_pred = model_cv.predict(X_test)
    
    return y_pred, y_pred_prob, model_cv


def RSDT(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro',random_state=1):
    params = {'max_depth':np.arange(1, 25, 1), 
              'criterion':np.array(['gini', 'entropy']),
              'min_samples_leaf':np.arange(1, 25, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=random_state)
    model = DecisionTreeClassifier(random_state= random_state)
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring=metric, n_jobs = -1, n_iter=50,random_state=random_state)

    model_cv.fit(X_train, y_train)
    y_pred_prob = model_cv.predict_proba(X_test)
    y_pred = model_cv.predict(X_test)
    
    return y_pred, y_pred_prob, model_cv

def RSDT_synclass(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro',random_state=1):
    params = {'max_depth':np.arange(1, 25, 1), 
              'criterion':np.array(['gini', 'entropy']),
              'min_samples_leaf':np.arange(1, 25, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
    model = DecisionTreeClassifier(random_state=random_state)
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring=metric, n_jobs = -1, n_iter=50,random_state=random_state)
    
    #print(X_train.shape)
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    #print(X_train.shape)
    model_cv.fit(X_train, y_train)
    
    y_pred_prob = model_cv.predict_proba(X_test)
    y_pred = model_cv.predict(X_test)

    return y_pred, y_pred_prob, model_cv


def _RSSVM(X_train, y_train, X_test, y_test, prob=False):
    """
    params = {'kernel': np.array(['linear', 'poly', 'rbf', 'sigmoid']), 
              'gamma':np.arange(0.01, 0.1, 0.01),
              'degree':np.arange(2, 5, 1),
              'coef0':np.arange(0, 2, 0.1)}    
    """

    # List of C values
    C_range = np.logspace(-10, 10, 21)
    
    # List of gamma values
    gamma_range = np.logspace(-10, 10, 21)
    
    # Define the search space
    param_grid = { 
        # Regularization parameter.
        "C": C_range,
        # Kernel type
        "kernel": ['rbf', 'poly'],
        # Gamma is the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        "gamma": gamma_range
        }

    
   
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = SVC(probability=True)
    model_cv = RandomizedSearchCV(model, param_grid, cv = 10, scoring='f1_macro', n_jobs = -1, n_iter=50)

    model_cv.fit(X_train, y_train)
    
    y_pred_prob = model_cv.predict_proba(X_test)
    y_pred = model_cv.predict(X_test)
    
    return y_pred, y_pred_prob, model_cv


def RSSVM(X_train, y_train, X_test, y_test, metric='f1',probability = True,n_jobs = -1):
    params = {'kernel': np.array(['linear', 'poly', 'rbf', 'sigmoid']), 
              'gamma':np.arange(0.01, 0.1, 0.01),
              'degree':np.arange(2, 5, 1),
              'coef0':np.arange(0, 2, 0.1)}    
   
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    model = SVC(probability=probability)
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring=metric, n_jobs = n_jobs, n_iter=50)
    model_cv.fit(X_train, y_train)
    
    y_pred_prob = []
    y_pred = []

    if probability:
        y_pred_prob = model_cv.predict_proba(X_test)
        y_pred = model_cv.predict(X_test)
    
    return y_pred, y_pred_prob, model_cv
    

def SVM_class_weight(X_train, y_train, X_test, y_test, prob=False,n_jobs = -1):
    
    #model = SVC(class_weight='balanced', gamma='scale')

    model = SVC(gamma='scale')
    
    params = {'kernel': np.array(['linear', 'poly', 'rbf', 'sigmoid']), 
              'gamma':np.arange(0.01, 0.1, 0.01),
              'degree':np.arange(2, 5, 1),
              'coef0':np.arange(0, 2, 0.1),
              'class_weight': [{-1: 0.9, 1: 0.1}, {-1: 0.8, 1: 0.2}, {-1: 0.7, 1: 0.3}, {-1: 0.6, 1: 0.4}, {-1: 0.5, 1: 0.5},
                               {-1: 0.8, 1: 0.7}, {-1: 0.7, 1: 0.6}, {-1: 0.6, 1: 0.5}]}

    # define grid
    #balance = [{-1:0.7, 1:0.6}, {-1:0.6, 1:0.5}, {-1:0.6, 1:0.4}, {-1:0.9, 1:0.6}]
    
    #balance = [{'Yes':1.0, 'No':0.5}]
    #param_grid = dict(class_weight=balance)

    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define grid search
    #grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=cv, scoring='f1_macro')

    grid = RandomizedSearchCV(model, params, cv = cv, scoring='f1_macro', n_jobs =n_jobs , n_iter=50)

    # execute the grid search
    grid_result = grid.fit(X_train, y_train)

    y_pred = grid_result.predict(X_test)

    return grid_result, y_pred


def SVM_class_weight_hiperopt(X_train, y_train, X_test, y_test, prob=False):
    '''
    SVM weight with parameter tuning with Hyperopt (Bayesian optimization)
    '''
    model = SVC(gamma='scale')

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    cw = [{-1: 0.9, 1: 0.1}, {-1: 0.8, 1: 0.2}, {-1: 0.7, 1: 0.3}, 
          {-1: 0.6, 1: 0.4}, {-1: 0.5, 1: 0.5}, {-1: 0.8, 1: 0.7}, 
          {-1: 0.7, 1: 0.6}, {-1: 0.6, 1: 0.5}]
    
    params = {'kernel': hp.choice('kernel', kernels), 
              'gamma': hp.uniform('gamma', 0, 20.0),
              'degree': hp.choice('degree', np.arange(2, 5, 1)),
              'coef0': hp.choice('coef0', np.arange(0, 2, 0.1)),
              'class_weight': hp.pchoice('class_weight', cw)
              }

    space = {'data': {'x': X_train, 'y': y_train},
            'model': model,
            'params': params,
            }

    trials = Trials()
    
    best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)

    log.info(f'Best params: {best}')

    best_model = SVC()
    best_model.set_params(**best)
    best_model.set_params(kernel=kernels[best['kernel']])
    best_model.set_params(class_weight=cw[best['class_weight']])

    best_model.fit(X_train, y_train)

    if prob:
        y_pred = best_model.predict_proba(X_test)
    else:
        y_pred = best_model.predict(X_test)
    
    return best_model, y_pred


def stacking_svm_dt(x_train, y_train, x_test, y_test, prob=False):

    estimators = [('rf', RandomForestClassifier()),
              ('svm', SVC())]
    
    params = {'rf__n_estimators': np.arange(1, 25, 1), 
            'rf__max_depth':np.arange(1, 25, 1),

            'svm__kernel': np.array(['linear', 'poly', 'rbf', 'sigmoid']), 
            'svm__gamma':np.arange(0.01, 0.1, 0.01),
            'svm__degree':np.arange(2, 5, 1),
            'svm__coef0':np.arange(0, 2, 0.1)
            }

    ## Run Random serach to get best params for models
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=2)

    sclf = StackingClassifier(estimators=estimators, 
                          final_estimator=LogisticRegression(penalty='none'), 
                          stack_method='predict')

    model_cv = RandomizedSearchCV(sclf, params, n_iter=1000, scoring='f1_macro', n_jobs = -1, cv=cv, random_state=1)
    model_cv.fit(x_train, y_train)
            
    y_pred = model_cv.predict(x_test)

    return model_cv, y_pred


def RS_stacking_classifier(x_train, y_train, x_test, y_test, prob=False,estimators=[('rf', RandomForestClassifier()),
    ('knn',  KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier())
    ], params = {'rf__n_estimators': np.arange(1, 25, 1), 
            'rf__max_depth':np.arange(1, 25, 1),

            'knn__n_neighbors': np.arange(1, 25, 1), 
            'knn__weights': np.array(['uniform', 'distance']),
            'knn__p': np.arange(1, 4, 1),

            'dt__max_depth':np.arange(1, 25, 1), 
            'dt__criterion':np.array(['gini', 'entropy']),
            'dt__min_samples_leaf':np.arange(1, 25, 1)}, meta_classifer=LogisticRegression(), random_state=1):
    '''
    Random Forest - ok
    Decision Tree -- ok
    KNN -- ok
    One Class SVM -- ok
    LOF - ok
    XGB
    '''


    # estimators = [('rf', RandomForestClassifier()),
    #               ('knn',  KNeighborsClassifier()),
    #               ('dt', DecisionTreeClassifier()), 
    #               #('oneclass', OneClassSVM(max_iter = 500)), 
    #               #("lof", LocalOutlierFactor(novelty=True))
    #               ]
    
    # params = {'rf__n_estimators': np.arange(1, 25, 1), 
    #         'rf__max_depth':np.arange(1, 25, 1),

    #         'knn__n_neighbors': np.arange(1, 25, 1), 
    #         'knn__weights': np.array(['uniform', 'distance']),
    #         'knn__p': np.arange(1, 4, 1),

    #         'dt__max_depth':np.arange(1, 25, 1), 
    #         'dt__criterion':np.array(['gini', 'entropy']),
    #         'dt__min_samples_leaf':np.arange(1, 25, 1), 

    #         #"oneclass__kernel": ['rbf', 'poly', 'sigmoid'],
    #         #"oneclass__gamma": np.arange(0.1, 5, 0.3),
    #         #"oneclass__coef0": np.arange(0, 5, 0.2),
    #         #"oneclass__degree": np.arange(1, 8, 1), 

    #         #'lof__algorithm':['ball_tree', 'kd_tree', 'brute'], 
    #         #'lof__n_neighbors':[5, 10, 15, 20, 25, 30, 35, 40, 45], 
    #         #'lof__contamination': [0.35, 0.40, 0.45, 0.5], 
    #         #'lof__leaf_size': [5, 10, 20, 30, 40],
    #         #'lof__metric': ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'], 
    #         #'lof__p':[1,2,3,4,5]
    #         }

    # Run Random search to get best params for models
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    
    # Use predict stack method to give the y_pred and calculate the scores after
    sclf = StackingClassifier(estimators=estimators, 
                          final_estimator=meta_classifer, 
                          stack_method='predict')

    model_cv = RandomizedSearchCV(sclf, params, n_iter=500, scoring='f1_macro', n_jobs = -1, cv=cv, random_state=random_state)
    model_cv.fit(x_train, y_train)
            
    y_pred = model_cv.predict(x_test)
    y_prob = model_cv.predict_proba(x_test)
    
    return y_pred, y_prob, model_cv

def RS_stacking_classifier_synclass(x_train, y_train, x_test, y_test, prob=False,metric="f1_macro",random_state=1,estimators=[
    ('rf', RandomForestClassifier()),
    ('knn',  KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier())
    ], params = {'rf__n_estimators': np.arange(1, 25, 1), 
            'rf__max_depth':np.arange(1, 25, 1),

            'knn__n_neighbors': np.arange(1, 25, 1), 
            'knn__weights': np.array(['uniform', 'distance']),
            'knn__p': np.arange(1, 4, 1),

            'dt__max_depth':np.arange(1, 25, 1), 
            'dt__criterion':np.array(['gini', 'entropy']),
            'dt__min_samples_leaf':np.arange(1, 25, 1)}, meta_classifer=LogisticRegression()):
            
    '''
    Random Forest - ok
    Decision Tree -- ok
    KNN -- ok
    One Class SVM -- ok
    LOF - ok
    XGB
    '''


    # estimators = [('rf', RandomForestClassifier()),
    #               ('knn',  KNeighborsClassifier()),
    #               ('dt', DecisionTreeClassifier()), 
    #               #('oneclass', OneClassSVM(max_iter = 500)), 
    #               #("lof", LocalOutlierFactor(novelty=True))
    #               ]
    
    # params = {'rf__n_estimators': np.arange(1, 25, 1), 
    #         'rf__max_depth':np.arange(1, 25, 1),

    #         'knn__n_neighbors': np.arange(1, 25, 1), 
    #         'knn__weights': np.array(['uniform', 'distance']),
    #         'knn__p': np.arange(1, 4, 1),

    #         'dt__max_depth':np.arange(1, 25, 1), 
    #         'dt__criterion':np.array(['gini', 'entropy']),
    #         'dt__min_samples_leaf':np.arange(1, 25, 1), 

    #         #"oneclass__kernel": ['rbf', 'poly', 'sigmoid'],
    #         #"oneclass__gamma": np.arange(0.1, 5, 0.3),
    #         #"oneclass__coef0": np.arange(0, 5, 0.2),
    #         #"oneclass__degree": np.arange(1, 8, 1), 

    #         #'lof__algorithm':['ball_tree', 'kd_tree', 'brute'], 
    #         #'lof__n_neighbors':[5, 10, 15, 20, 25, 30, 35, 40, 45], 
    #         #'lof__contamination': [0.35, 0.40, 0.45, 0.5], 
    #         #'lof__leaf_size': [5, 10, 20, 30, 40],
    #         #'lof__metric': ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'], 
    #         #'lof__p':[1,2,3,4,5]
    #         }

    # Run Random search to get best params for models
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    
    # Use predict stack method to give the y_pred and calculate the scores after
    sclf = StackingClassifier(estimators=estimators, 
                          final_estimator=meta_classifer, 
                          stack_method='predict',n_jobs=-1)

    model_cv = RandomizedSearchCV(sclf, params, n_iter=500, scoring=metric, n_jobs = -1, cv=cv, random_state=random_state)
    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)
    model_cv.fit(x_train, y_train)
            
    y_pred = model_cv.predict(x_test)
    y_prob = model_cv.predict_proba(x_test)
    
    return y_pred, y_prob, model_cv

def SVM_hiperopt(X_train, y_train, X_test, y_test, prob=False, metric='f1_macro'):
    '''
    SVM  with parameter tuning with Hyperopt (Bayesian optimization)
    '''
    model = SVC(gamma='scale')

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    
    params = {'kernel': hp.choice('kernel', kernels), 
              'gamma': hp.uniform('gamma', 0, 20.0),
              'degree': hp.choice('degree', np.arange(2, 5, 1)),
              'coef0': hp.choice('coef0', np.arange(0, 2, 0.1)),
              }

    space = {'data': {'x': X_train, 'y': y_train},
            'model': model,
            'params': params,
            }

    trials = Trials()
    
    best = fmin(f, space, algo=tpe.suggest, max_evals=50, trials=trials)

    log.info(f'Best params: {best}')

    best_model = SVC(probability=True)
    best_model.set_params(**best)
    best_model.set_params(kernel=kernels[best['kernel']])

    best_model.fit(X_train, y_train)

    y_pred_prob = best_model.predict_proba(X_test)
    y_pred = best_model.predict(X_test)
    
    return y_pred, y_pred_prob, best_model