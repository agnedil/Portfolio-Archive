# Best article explaining calibration and how to calibrate: https://towardsdatascience.com/classifier-calibration-7d0be1e05452
# How to implement calibration for NNs 1: https://wttech.blog/blog/2021/a-guide-to-model-calibration/
# How to implement calibration for NNs 2: https://github.com/markus93/NN_calibration  (complicated)
# Calibration formulas: https://medium.com/analytics-vidhya/calibration-in-machine-learning-e7972ac93555

# ROC curve optimal threshold formula taken from here: https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python


import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_curve, roc_auc_score


def calibration_plot(y_true, y_pred, name='Current Model'):
    '''
        Display model calibration plot.
        If normalize=True, the smallest value in y_pred is linearly mapped onto 0 and the largest one onto 1 (default=False).
        Strategy: a) 'uniform' - bins have identical widths; b) 'quantile' - bins have the same number of samples and depend on y_pred (default='uniform')

        Another method:
            disp = sklearn.calibration.CalibrationDisplay.from_predictions(y_true, y_pred)
            plt.show()
    '''
    x, y  = calibration_curve(y_true, y_pred, n_bins=10, normalize=False, strategy='uniform')
    score = brier_score_loss(y_true, y_pred)

    plt.plot([0, 1], [0, 1], linestyle='--', label='Ideally Calibrated')
    plt.plot(y, x, "s-", marker = '.', label=f'{name} (Brier={round(score, 4)})')
    plt.title('Calibration Plot')
    plt.legend(loc='lower right')
    plt.xlabel('Mean predicted value (per bin)')
    plt.ylabel('Fraction of positives (per bin)')
    plt.show()


def roc_plot(y_true, y_pred, name='Current Model'):
    '''
        Display a ROC curve with AUC score
    '''
    fpr, tpr, thresh = roc_curve(y_true, y_pred, pos_label=1)
    optim_thresh     = round(thresh[ np.argmax(tpr - fpr) ], 4)
    auc_score        = round(roc_auc_score(y_true, y_pred), 4)

    plt.plot([0, 1], [0, 1], linestyle='--', color='tab:blue',)
    plt.plot(fpr, tpr, color='tab:orange',)
    plt.title(f"ROC Curve (AUC Score={auc_score:.4f}, optimal threshold={optim_thresh:.4f}?)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
