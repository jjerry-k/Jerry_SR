import numpy as np
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mutual_info_score as mi
from sklearn.metrics import normalized_mutual_info_score as nmi

def MAE(y_true, y_pred):
    """
    y_true : (Batch, Height, Width, Slices)
    y_pred : (Batch, Height, Width, Slices)
    """
    return np.mean(np.abs(y_true-y_pred), axis=(1, 2))

def MSE(y_true, y_pred):
    """
    y_true : (Batch, Height, Width, Slices)
    y_pred : (Batch, Height, Width, Slices)
    """
    return np.mean(np.square(y_true-y_pred), axis=(1, 2))

def RMSE(y_true, y_pred):
    """
    y_true : (Batch, Height, Width, Slices)
    y_pred : (Batch, Height, Width, Slices)
    """
    return np.sqrt(MSE(y_true, y_pred))

def RMSPE(y_true, y_pred):
    """
    y_true : (Batch, Height, Width, Slices)
    y_pred : (Batch, Height, Width, Slices)
    """
    return np.sqrt(np.mean(np.square((y_true - y_pred + 1)/(y_true+ 1)), axis=(1, 2)))

def PSNR(y_true, y_pred):
    '''
    y_true : (Batch, Height, Width, Slices)
    y_pred : (Batch, Height, Width, Slices)
    d_r : dynamic range
    '''
    mse = MSE((y_true/y_true.max(axis=(1, 2), keepdims=True)*255).astype('uint8'), (y_pred/y_pred.max(axis=(1, 2), keepdims=True)*255).astype('uint8'))
    #d_r = y_true.max(axis=(1,2)) - y_true.min(axis=(1,2))
    return 10 * np.log10((255**2)/mse)

def SSIM(y_true, y_pred):
    b, _, _, ch = y_true.shape
    output = np.zeros((b, ch))
    for b_idx in range(b):
        for ch_idx in range(ch):
            true_max = y_true[b_idx, ..., ch_idx].max()
            pred_vmin = y_pred[b_idx, ..., ch_idx].max()
            output[b_idx, ch_idx] = ssim((y_true[b_idx, ..., ch_idx]/true_max*255).astype('uint8'), (y_pred[b_idx, ..., ch_idx]/pred_vmin*255).astype('uint8'), data_range=255)
            
    return output

def MI(y_true, y_pred):
    b, _, _, ch = y_true.shape
    output = np.zeros((b, ch))
    for b_idx in range(b):
        for ch_idx in range(ch):
            true_max = y_true[b_idx, ..., ch_idx].max()
            pred_vmin = y_pred[b_idx, ..., ch_idx].max()
            output[b_idx, ch_idx] = mi((y_true[b_idx, ..., ch_idx]/true_max*255).astype('uint8').ravel(), (y_pred[b_idx, ..., ch_idx]/pred_vmin*255).astype('uint8').ravel())
    return output

def NMI(y_true, y_pred):
    b, _, _, ch = y_true.shape
    output = np.zeros((b, ch))
    for b_idx in range(b):
        for ch_idx in range(ch):
            true_max = y_true[b_idx, ..., ch_idx].max()
            pred_vmin = y_pred[b_idx, ..., ch_idx].max()
            output[b_idx, ch_idx] = nmi((y_true[b_idx, ..., ch_idx]/true_max*255).astype('uint8').ravel(), (y_pred[b_idx, ..., ch_idx]/pred_vmin*255).astype('uint8').ravel())
    return output