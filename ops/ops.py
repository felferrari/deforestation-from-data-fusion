import json
import ee
import time
import numpy as np
import os
from skimage.util import view_as_windows, crop
from sklearn.metrics import confusion_matrix
import math as m
import sys
from osgeo import gdal_array

def load_json(fp):
    with open(fp) as f:
        return json.load(f)
    
def save_json(dict_, fp):
    with open(fp, 'w') as f:
        json.dump(dict_, f, indent=4)
    
    
def load_opt_image(patch):
    # Read tiff Image
    #print (patch)
    #img_tif = TIFF.open(patch)
    #img = img_tif.read_image()
    img = gdal_array.LoadFile(patch)
    return np.moveaxis(img, 0, -1)

def load_label_image(patch):
    img = gdal_array.LoadFile(patch)
    return img

def load_SAR_image(patch):
    '''Function to read SAR images'''
    db_img = gdal_array.LoadFile(patch)
    temp_dn_img = 10**(db_img/10)
    temp_dn_img[temp_dn_img>1] = 1
    return np.moveaxis(temp_dn_img, 0, -1)

def load_SAR_DN_image(patch):
    '''Function to read SAR images'''
    im = gdal_array.LoadFile(patch)
    return np.expand_dims(im, axis=-1)


def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)]=0 # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(),bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        max_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<uth])])/100
        min_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<bth])])/100
        img[:,:, band][img[:,:, band]>max_value] = max_value
        img[:,:, band][img[:,:, band]<min_value] = min_value
    return img

def normalize(data):
    st = np.load(os.path.join("statistics.npz"))
    opt_mean = st['opt_mean']
    opt_std = st['opt_std']
    sar_mean = st['sar_mean']
    sar_std = st['sar_std']
    return (
        (data[0]-opt_mean)/opt_std,
        (data[1]-sar_mean)/sar_std,
    )

def patchs_full_gen(image, patch_size, patch_step):
    shape = image.shape
    d_patch_size = int((patch_size - patch_step)/2)
    n_l = m.ceil(shape[0]/patch_step)
    n_c = m.ceil(shape[1]/patch_step)
    l = n_l*patch_step + (patch_size - patch_step)
    c = n_c*patch_step + (patch_size - patch_step)


    #extra_l = patch_step# - int(shape[0]%patch_step)
    #extra_c = patch_step# - int(shape[1]%patch_step)
    if len(shape) == 3:
        pad_shape = (
            (d_patch_size, l-shape[0]-d_patch_size),
            (d_patch_size, c-shape[1]-d_patch_size),
            (0,0)
        )
        return view_as_windows(
            np.pad(image, pad_shape , 'reflect' ), 
            (patch_size, patch_size, shape[-1]), 
            patch_step).squeeze()
    elif len(shape) == 2:
        pad_shape = (
            (d_patch_size, l-shape[0]-d_patch_size),
            (d_patch_size, c-shape[1]-d_patch_size)
        )
        return view_as_windows(
            np.pad(image, pad_shape , 'reflect' ), 
            (patch_size, patch_size), 
            patch_step).squeeze()

def predict_from_patches(patches, patch_size, patch_step, model, final_shape):
    
    d_patch_size = int((patch_size - patch_step)/2)
    crop_shape = (
        (0,0),
        (d_patch_size, d_patch_size),
        (d_patch_size, d_patch_size),
        (0,0)
    )
    
    pred = None
    
    for pi in patches:
        if pred is None:
            pred = np.column_stack(crop(model.predict(pi), crop_shape))
        else:
            pred = np.concatenate([
                pred,
                np.column_stack(crop(model.predict(pi), crop_shape))
            ], axis= 0)

    return pred[:final_shape[0], :final_shape[1], :]

def rebuild_from_patches(patches, final_shape):
    pred = None
    
    for pi in patches:
        if pred is None:
            pred = np.column_stack(pi)
        else:
            pred = np.concatenate([
                pred,
                np.column_stack(pi)
            ], axis= 0)
            
    return pred[:final_shape[0], :final_shape[1], :]

def predict_from_patches_multi(patches, patch_size, patch_step, model, final_shape):
    
    d_patch_size = int((patch_size - patch_step)/2)
    crop_shape = (
        (0,0),
        (d_patch_size, d_patch_size),
        (d_patch_size, d_patch_size),
        (0,0)
    )
    
    pred_0 = None
    pred_1 = None
    
    for pi in patches:
        if pred_0 is None:
            pred_0 = np.column_stack(crop(model.predict(pi)[0], crop_shape))
            pred_1 = np.column_stack(crop(model.predict(pi)[1], crop_shape))
        else:
            pred_0 = np.concatenate([
                pred_0,
                np.column_stack(crop(model.predict(pi)[0], crop_shape))
            ], axis= 0)
            pred_1 = np.concatenate([
                pred_1,
                np.column_stack(crop(model.predict(pi)[1], crop_shape))
            ], axis= 0)

    return pred_0[:final_shape[0], :final_shape[1], :], pred_1[:final_shape[0], :final_shape[1], :]


def predict_from_patches_multi_in_multi_out(opt_patches, sar_patches, patch_size, patch_step, model, final_shape):
    
    d_patch_size = int((patch_size - patch_step)/2)
    crop_shape = (
        (0,0),
        (d_patch_size, d_patch_size),
        (d_patch_size, d_patch_size),
        (0,0)
    )
    
    pred_0 = None
    pred_1 = None
    pred_2 = None
    
    for opt_pi, sar_pi in zip(opt_patches, sar_patches):
        if pred_0 is None:
            pred_0 = np.column_stack(crop(model.predict((opt_pi, sar_pi))[0], crop_shape))
            pred_1 = np.column_stack(crop(model.predict((opt_pi, sar_pi))[1], crop_shape))
            pred_2 = np.column_stack(crop(model.predict((opt_pi, sar_pi))[2], crop_shape))
        else:
            pred_0 = np.concatenate([
                pred_0,
                np.column_stack(crop(model.predict((opt_pi, sar_pi))[0], crop_shape))
            ], axis= 0)
            pred_1 = np.concatenate([
                pred_1,
                np.column_stack(crop(model.predict((opt_pi, sar_pi))[1], crop_shape))
            ], axis= 0)
            pred_2 = np.concatenate([
                pred_2,
                np.column_stack(crop(model.predict((opt_pi, sar_pi))[2], crop_shape))
            ], axis= 0)

    return pred_0[:final_shape[0], :final_shape[1], :], pred_1[:final_shape[0], :final_shape[1], :], pred_2[:final_shape[0], :final_shape[1], :]


def rebuild_from_patches2(patches, patch_size, patch_step, final_shape):
    
    d_patch_size = int((patch_size - patch_step)/2)
    crop_shape = (
        (0,0),
        (d_patch_size, d_patch_size),
        (d_patch_size, d_patch_size),
        (0,0)
    )
    
    pred = None
    
    for pi in patches:
        if pred is None:
            pred = np.column_stack(crop(pi, crop_shape))
        else:
            pred = np.concatenate([
                pred,
                np.column_stack(crop(pi, crop_shape))
            ], axis= 0)

    return pred[:final_shape[0], :final_shape[1], :]



def precision_recall(lim, pred, label, mask_consider):
    print(lim)
    pred_t = np.zeros_like(pred, dtype=np.byte)
    pred_t[pred>=lim] = 1

    pred_t = pred_t*mask_consider
    #pred_t = area_opening(pred_t, 625)

    tn, fp, fn, tp = confusion_matrix(pred_t[mask_consider==1].flatten(), label[:,:,1][mask_consider==1].flatten()).ravel()

    precision = tp/(tp+fp)
    recall =  tp/(tp+fn)
    acc = (tp+tn)/(tp+fp+fn+tn)

    if np.isnan(precision):
        precision = 1-recall
    if np.isnan(recall):
        recall = 1-precision

    return (precision, recall, acc)


def create_exps_paths(exp_n):
    exps_path = 'exps'

    exp_path = os.path.join(exps_path, f'exp_{exp_n}')
    models_path = os.path.join(exp_path, 'models')

    results_path = os.path.join(exp_path, 'results')
    predictions_path = os.path.join(results_path, 'predictions')
    visual_path = os.path.join(results_path, 'visual')

    logs_path = os.path.join(exp_path, 'logs')

    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    return exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path

def load_exp(exp_n = None):
    if exp_n is None:
        if len(sys.argv)==1:
            return load_json(os.path.join('conf', f'exps.json'))
        else:
            return load_json(os.path.join('conf', f'exps {sys.argv[1]}.json'))
    else:
        return load_json(os.path.join('conf', f'exps {exp_n}.json'))
    


