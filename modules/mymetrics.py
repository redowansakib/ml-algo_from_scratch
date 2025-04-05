from modules.mylinalg import *
from modules.mystats import *

def r2_score(y_true, y_pred):

    y_true,y_pred = to2dim(y_true), to2dim(y_pred)

    if len(y_true) != len(y_pred):
        raise Exception('Both y_true and y_pred must have same length')


    RSS = sum([(y_true[i][0]-y_pred[i][0])**2 for i in range(len(y_true))])
    y_bar = mean(y_true)
    TSS = sum([(y_true[i][0]-y_bar)**2 for i in range(len(y_true))])

    r2 = 1-(RSS/TSS)

    return r2