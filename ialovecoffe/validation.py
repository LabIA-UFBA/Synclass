from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

# from synclass.sfc import roc_calc_viz_pred
def positive_negative_rate(y_test, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensibivity = tp / (tp+fn)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    return round(sensibivity, 5), round(specificity, 5), round(accuracy,5) 


def accuracy_per_class(y_test, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc_class_1 = tp / (tp + fp)
    acc_class_2 = tn / (tn + fn)

    # Calculates tpr and fpr
    #tpr =  tp/(tp + fn) # sensitivity - true positive rate
    #fpr = 1 - tn/(tn+fp) # 1-specificity - false positive rate
    
    return acc_class_1, acc_class_2

def roc_calc_viz_pred(y_true, y_pred):
    viz = RocCurveDisplay.from_predictions(
                            y_true,
                            y_pred
                        )

    return viz.fpr, viz.tpr, viz.roc_auc


def computer_scores_roc(y_test, y_pred, y_pred_prob, names=False):
    
    sen, spe,acc = positive_negative_rate(y_test, y_pred)
    f1 = round(f1_score(y_test, y_pred, average='macro'), 5)
    roc = round(roc_auc_score(y_test, y_pred, average='weighted'), 5)
    jac = round(jaccard_score(y_test, y_pred, average='weighted'), 5)
    fmi = round(fowlkes_mallows_score(y_test, y_pred), 5)
    mcc = round(matthews_corrcoef(y_test, y_pred), 5)
    #fpr, tpr, thresh = roc_curve(y_test, y_pred_prob[:,1], pos_label=1)
    fpr,tpr,auc = roc_calc_viz_pred(y_test, y_pred)
    acc_class_1,acc_class_2 = accuracy_per_class(y_test,y_pred)
    if names:
        return {'SEN': sen, 'SPE': spe, 'F1':f1, 
                'ROC': roc, 'IOU': jac, 'FMI': fmi, 'MCC': mcc}

    return sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr , auc,acc_class_1,acc_class_2,acc #, thresh


def computer_scores(y_test, y_pred, names=False):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = (tp + tn)/(tp + tn + fp + fn)
    sen, spe = positive_negative_rate(y_test, y_pred)
    f1 = round(f1_score(y_test, y_pred, average='binary'), 5)
    roc = round(roc_auc_score(y_test, y_pred, average='weighted'), 5)
    jac = round(jaccard_score(y_test, y_pred, average='weighted'), 5)
    fmi = round(fowlkes_mallows_score(y_test, y_pred), 5)
    mcc = round(matthews_corrcoef(y_test, y_pred), 5)

    if names:
        return {'SEN': sen, 'SPE': spe, 'F1':f1, 
                'ROC': roc, 'IOU': jac, 'FMI': fmi, 'MCC': mcc, "ACC":acc}

    return sen, spe, f1, roc, jac, fmi, mcc, acc

def computer_scores_outlier(y_test, y_pred, names=False):
    sen, spe = positive_negative_rate(y_test, y_pred)
    f1 = round(f1_score(y_test, y_pred, average='macro'), 5)
    roc = round(roc_auc_score(y_test, y_pred, average='weighted'), 5)
    jac = round(jaccard_score(y_test, y_pred, average='weighted'), 5)
    fmi = round(fowlkes_mallows_score(y_test, y_pred), 5)
    mcc = round(matthews_corrcoef(y_test, y_pred), 5)
    fpr, tpr, thresh = [], [], []

    if names:
        return {'SEN': sen, 'SPE': spe, 'F1':f1, 
                'ROC': roc, 'IOU': jac, 'FMI': fmi, 'MCC': mcc}

    return sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh