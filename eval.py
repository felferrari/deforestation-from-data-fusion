import numpy as np
import os
from ops.ops import load_json, create_exps_paths, load_exp
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import time
from sklearn.metrics import confusion_matrix
from skimage.morphology import area_opening
import fiona
from mpl_toolkits.axes_grid1 import make_axes_locatable



def eval_model_metrics(m_idx):
    threshold = 0.5
    t0 = time.perf_counter()

    shp = load_json(os.path.join('conf', 'shp.json'))
    conf = load_json(os.path.join('conf', 'conf.json'))
    exp = load_exp()

    exp_n = exp['exp_n']

    exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path = create_exps_paths(exp_n)

    log_file = os.path.join(exp_path, f'eval.txt')

    logging.basicConfig(
        filename = log_file, 
        level = logging.INFO,
        filemode = 'a',
        format = '%(asctime)s - %(message)s', 
        datefmt = '%d-%b-%y %H:%M:%S'
        )

    test_tiles_path = os.path.join('img', 'test_tiles')

    img_source = conf['img_source']
    grid_shp = shp[f'shp_download_{img_source}']
    grid_save = os.path.join('shp', f"{grid_shp}.shp")

    test_feat_ids = []
    with fiona.open(grid_save) as grid:
        for feat in grid:
            feat_id = int(feat['properties']['id'])
            if feat['properties']['dataset'] == 2: # test dataset
                test_feat_ids.append(feat_id)
    cmap_lim = conf['cmap_lim']
    model_metrics = np.zeros((4, 4))

    for feat_id in test_feat_ids:
        fz = np.load(os.path.join(test_tiles_path, f'{feat_id}.npz'))
        label = fz['label'].squeeze()

        cmap_max =  fz['cmap'].max(axis=-1)

        pred_prob = np.load(os.path.join(predictions_path, f'{feat_id}_fus_{m_idx}.npy'))[:,:,1]
        #pred_label = np.argmax(pred_prob, axis=-1)
        pred = np.zeros_like(pred_prob)
        pred[pred_prob >= threshold] = 1
        #pred[pred_label == 1] = 1
        #pred_size_removed = pred -  area_opening(pred, 625)
        #label[pred_size_removed == 1] = 2

        cm_idx = []
        cm_idx.append(cmap_max<=cmap_lim[0])
        cm_idx.append(np.logical_and(cmap_max>cmap_lim[0], cmap_max<=cmap_lim[1]))
        cm_idx.append(cmap_max>cmap_lim[1])
        cm_idx.append(cmap_max>=0)


        for cmap_i in range(len(cm_idx)):
            model_metrics[0, cmap_i]+= np.logical_and(pred[cm_idx[cmap_i]]==1, label[cm_idx[cmap_i]]==1).sum() #tp
            model_metrics[1, cmap_i]+= np.logical_and(pred[cm_idx[cmap_i]]==0, label[cm_idx[cmap_i]]==0).sum() #tn
            model_metrics[2, cmap_i]+= np.logical_and(pred[cm_idx[cmap_i]]==1, label[cm_idx[cmap_i]]==0).sum() #fp
            model_metrics[3, cmap_i]+= np.logical_and(pred[cm_idx[cmap_i]]==0, label[cm_idx[cmap_i]]==1).sum() #fn

    logging.info(f'Evaluation of model #{m_idx} consumed {(time.perf_counter() - t0)/60:.2f} mins')

    return model_metrics



def eval_feat_curves(threshold):

    t0 = time.perf_counter()

    shp = load_json(os.path.join('conf', 'shp.json'))
    
    exp = load_exp()

    exp_n = exp['exp_n']

    exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path = create_exps_paths(exp_n)

    log_file = os.path.join(exp_path, f'eval.txt')

    logging.basicConfig(
        filename = log_file, 
        level = logging.INFO,
        filemode = 'a',
        format = '%(asctime)s - %(message)s', 
        datefmt = '%d-%b-%y %H:%M:%S'
        )

    train_patches_path = os.path.join('img', 'patches')

    conf = load_json(os.path.join('conf', 'conf.json'))
    img_source = conf['img_source']
    n_cmap_layers = conf['n_cmap_layers']


    grid_shp = shp[f'shp_download_{img_source}']
    grid_save = os.path.join('shp', f"{grid_shp}.shp")

    test_feat_ids = []
    with fiona.open(grid_save) as grid:
        for feat in grid:
            feat_id = int(feat['properties']['id'])
            if feat['properties']['dataset'] == 2: # test dataset
                test_feat_ids.append(feat_id)
    cmap_lim = conf['cmap_lim']
    curves = np.zeros((4, 4))

    for feat_id in test_feat_ids:
        fz = np.load(os.path.join(train_patches_path, f'{feat_id}.npz'))
        shape = fz['shape']
        shape = (shape[0], shape[1])
        label = fz['label'].reshape(shape)

        cmap_max =  fz['cmap'].reshape(shape+(n_cmap_layers,)).max(axis=-1)

        pred_prob = np.load(os.path.join(predictions_path, f'{feat_id}_fus.npy'))[:,:,1]
        pred = np.zeros_like(pred_prob)
        pred[pred_prob >= threshold] = 1

        #pred_size_removed = pred -  area_opening(pred, 625)
        #label[pred_size_removed == 1] = 2
        pred_red = area_opening(pred, 625)
        label[(pred - pred_red)==1] = 2

        cm_idx = []
        cm_idx.append(cmap_max<=cmap_lim[0])
        cm_idx.append(np.logical_and(cmap_max>cmap_lim[0], cmap_max<=cmap_lim[1]))
        cm_idx.append(cmap_max>cmap_lim[1])
        cm_idx.append(cmap_max>=0)


        for cmap_i in range(len(cm_idx)):
            curves[0, cmap_i]+= np.logical_and(pred[cm_idx[cmap_i]]==1, label[cm_idx[cmap_i]]==1).sum() #tp
            curves[1, cmap_i]+= np.logical_and(pred[cm_idx[cmap_i]]==0, label[cm_idx[cmap_i]]==0).sum() #tn
            curves[2, cmap_i]+= np.logical_and(pred[cm_idx[cmap_i]]==1, label[cm_idx[cmap_i]]==0).sum() #fp
            curves[3, cmap_i]+= np.logical_and(pred[cm_idx[cmap_i]]==0, label[cm_idx[cmap_i]]==1).sum() #fn

    logging.info(f'Evaluation of threshold {threshold:.3f} consumed {(time.perf_counter() - t0)/60:.2f} mins')

    return curves

def eval_feat(ha_min):

    t0 = time.perf_counter()

    shp = load_json(os.path.join('conf', 'shp.json'))
    
    exp = load_exp()

    exp_n = exp['exp_n']

    exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path = create_exps_paths(exp_n)

    log_file = os.path.join(exp_path, f'eval.txt')

    logging.basicConfig(
        filename = log_file, 
        level = logging.INFO,
        filemode = 'a',
        format = '%(asctime)s - %(message)s', 
        datefmt = '%d-%b-%y %H:%M:%S'
        )

    train_patches_path = os.path.join('img', 'patches')

    conf = load_json(os.path.join('conf', 'conf.json'))
    img_source = conf['img_source']
    n_cmap_layers = conf['n_cmap_layers']


    grid_shp = shp[f'shp_download_{img_source}']
    grid_save = os.path.join('shp', f"{grid_shp}.shp")

    test_feat_ids = []
    with fiona.open(grid_save) as grid:
        for feat in grid:
            feat_id = int(feat['properties']['id'])
            if feat['properties']['dataset'] == 2: # test dataset
                test_feat_ids.append(feat_id)
    cmap_lim = conf['cmap_lim']
    sums = np.zeros((4, 4))

    for feat_id in test_feat_ids:
        fz = np.load(os.path.join(train_patches_path, f'{feat_id}.npz'))
        shape = fz['shape']
        shape = (shape[0], shape[1])
        label = fz['label'].reshape(shape)

        cmap_max =  fz['cmap'].reshape(shape+(n_cmap_layers,)).max(axis=-1)

        pred_prob = np.load(os.path.join(predictions_path, f'{feat_id}_fus.npy'))[:,:,1]
        pred = np.zeros_like(pred_prob)
        pred[pred_prob >= 0.5] = 1

        #pred_size_removed = pred -  area_opening(pred, 625)
        #label[pred_size_removed == 1] = 2
        min_area = int(ha_min * 100)
        pred_rem = pred - area_opening(pred, min_area)
        #pred[pred_rem == 1] = 0
        label[pred_rem == 1] = 2

        label_pred = np.zeros_like(label)
        label_pred[label == 1] = 1
        label_rem = label_pred - area_opening(label_pred, min_area)
        #label[(label_pred - label_red)==1] = 2
        #pred[label_rem == 1] = 0
        label[label_rem == 1] = 2
        
        cm_idx = []
        cm_idx.append(cmap_max<=cmap_lim[0])
        cm_idx.append(np.logical_and(cmap_max>cmap_lim[0], cmap_max<=cmap_lim[1]))
        cm_idx.append(cmap_max>cmap_lim[1])
        cm_idx.append(cmap_max>=0)


        for cmap_i in range(len(cm_idx)):
            tp = np.logical_and(pred[cm_idx[cmap_i]]==1, label[cm_idx[cmap_i]]==1).sum() #tp
            tn = np.logical_and(pred[cm_idx[cmap_i]]==0, label[cm_idx[cmap_i]]==0).sum() #tn
            fp = np.logical_and(pred[cm_idx[cmap_i]]==1, label[cm_idx[cmap_i]]==0).sum() #fp
            fn = np.logical_and(pred[cm_idx[cmap_i]]==0, label[cm_idx[cmap_i]]==1).sum() #fn

            sums[0, cmap_i]+= tp
            sums[1, cmap_i]+= tn
            sums[2, cmap_i]+= fp
            sums[3, cmap_i]+= fn

    logging.info(f'Evaluation of area {ha_min:.3f} ha consumed {(time.perf_counter() - t0)/60:.2f} mins')

    return sums

def plot_curve(curve, file, ap):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.plot(curve[:,1], curve[:,0], 'b-', label = f'Precision (AP={ap:.4f})')
    ax.plot(curve[:,1], curve[:,2], 'r-', label = 'Accuracy')
    ax.set_xlim([0,1.01])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.legend(loc="upper right")
    #ax.set_axis_off()
    fig.savefig(file, transparent=True)
    plt.close()

def plot_ap_curves(precs, recalls, aps, file):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.plot(recalls[:,0], precs[:,0], 'r-', label = f'No Cloud (AP={aps[0]:.3f})')
    ax.plot(recalls[:,1], precs[:,1], 'g-', label = f'Medium Cloud (AP={aps[1]:.3f})')
    ax.plot(recalls[:,2], precs[:,2], 'b-', label = f'High Cloud (AP={aps[2]:.3f})')
    ax.plot(recalls[:,3], precs[:,3], 'k-', label = f'Global (AP={aps[3]:.3f})')
    ax.set_xlim([0,1.01])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.legend(loc="upper right")
    #ax.set_axis_off()
    fig.savefig(file, transparent=True)
    plt.close()

def plot_metrics_curves(curves, min_max, file):

    precision = curves[:, 0, :] / (curves[:,0,:] + curves[:,2,:])
    recall = curves[:, 0, :] / (curves[:,0,:] + curves[:,3,:])
    f1 = (2 * precision * recall)/(precision + recall)

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].plot(min_max, precision[:,0], 'r-', label = f'No Cloud')
    ax[0].plot(min_max, precision[:,1], 'g-', label = f'Medium Cloud')
    ax[0].plot(min_max, precision[:,2], 'b-', label = f'High Cloud')
    ax[0].plot(min_max, precision[:,3], 'k-', label = f'Global')
    ax[0].set_ylim([0,1])
    ax[0].set_xlabel('Minimum Area')
    ax[0].set_ylabel('Precision')

    ax[1].plot(min_max, recall[:,0], 'r-', label = f'No Cloud')
    ax[1].plot(min_max, recall[:,1], 'g-', label = f'Medium Cloud')
    ax[1].plot(min_max, recall[:,2], 'b-', label = f'High Cloud')
    ax[1].plot(min_max, recall[:,3], 'k-', label = f'Global')
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel('Minimum Area')
    ax[1].set_ylabel('Recall')

    ax[2].plot(min_max, f1[:,0], 'r-', label = f'No Cloud')
    ax[2].plot(min_max, f1[:,1], 'g-', label = f'Medium Cloud')
    ax[2].plot(min_max, f1[:,2], 'b-', label = f'High Cloud')
    ax[2].plot(min_max, f1[:,3], 'k-', label = f'Global')
    ax[2].set_ylim([0,1])
    ax[2].set_xlabel('Minimum Area')
    ax[2].set_ylabel('F1-Score')
    plt.legend(loc="lower right")
    fig.savefig(file, transparent=True)
    plt.close()




def plot_sample(pred, cmap, label, file):

    cmap_max = cmap.max(axis=-1)
    mask_consider = (1-label[:,:,2]).astype(np.byte)

    fig, ax = plt.subplots(2, 3, figsize=(30,20))
    im = ax[0,0].imshow(pred*mask_consider, cmap='plasma', vmin=0, vmax=1)
    ax[0,0].set_title('Prediction')
    ax[0,0].set_axis_off()
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Deforestation Probability", ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0], format = '%.1f')

    im = ax[0,1].imshow(label[:,:,1], cmap='plasma', vmin=0, vmax=1)
    ax[0,1].set_title('Ground Truth')
    ax[0,1].set_axis_off()
    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Deforestation Probability", ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0], format = '%.1f')

    
    error = (pred-label[:,:,1])*mask_consider
    im = ax[0,2].imshow(error, cmap='bwr', vmin=-1, vmax=1)
    ax[0,2].set_title('Error Map')
    ax[0,2].set_axis_off()
    divider = make_axes_locatable(ax[0,2])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Error: False Negative (-) or False Positive(+)", ticks = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0], format = '%.1f')

    im = ax[1,0].imshow(cmap[:,:,0], cmap='plasma', vmin=0, vmax=1)
    ax[1,0].set_title('Cloud Map Before')
    ax[1,0].set_axis_off()
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Cloud Probability", ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0], format = '%.1f')

    im = ax[1,1].imshow(cmap[:,:,1], cmap='plasma', vmin=0, vmax=1)
    ax[1,1].set_title('Cloud Map After')
    ax[1,1].set_axis_off()
    divider = make_axes_locatable(ax[1,1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Cloud Probability", ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0], format = '%.1f')

    im = ax[1,2].imshow(cmap_max, cmap='plasma', vmin=0, vmax=1)
    ax[1,2].set_title('Cloud Map Max')
    ax[1,2].set_axis_off()
    divider = make_axes_locatable(ax[1,2])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Cloud Probability", ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0], format = '%.1f')

    #ax.set_axis_off()
    fig.savefig(file, transparent=True)
    plt.close()

def eval_average_precision(pred, label, n_parts):
    limits = np.linspace(0,1,n_parts)
    mask_consider = (1-label[:,:,2]).astype(np.byte)

    curve = []

    for lim in limits:
        pred_t = np.zeros_like(pred, dtype=np.byte)
        pred_t[pred>=lim] = 1

        pred_t = pred_t*mask_consider
        #pred_t = area_opening(pred_t, 625)

        tn, fp, fn, tp = confusion_matrix(pred_t[mask_consider==1].flatten(), label[:,:,1][mask_consider==1].flatten()).ravel()

        precision = tp/(tp+fp)
        recall =  tp/(tp+fn)
        acc = (tp+tn)/(tp+fp+fn+tn)

        curve.append([precision, recall, acc])

    curve = np.array(curve)
    
    return curve



def correct_curve(curve):
    #remove nan
    last_prec = 0
    for i in range(curve.shape[0], 0):
        if np.isnan(curve[i, 0]):
            curve[i, 0] = last_prec
        else:
            last_rec = curve[i, 0]

    last_rec = 0
    for i in range(curve.shape[0]):
        if np.isnan(curve[i, 1]):
            curve[i, 1] = last_rec
        else:
            last_rec = curve[i, 1]

    curve = curve[curve[:,1].argsort()]

    for i in range(curve.shape[0]):
        curve[i, 0] = curve[i:, 0].max()

    return np.vstack([[curve[0,0], 0, curve[0,2]], curve])

def eval_ap(curve):
    d_rec = curve[1:,1] - curve[:-1,1]
    return (d_rec*curve[1:,0]).sum()


def eval_metrics_clouds(curves):
    precs = curves[:, 0, :] / (curves[:, 0, :] + curves[:, 2, :])
    #clean precision
    for cm_idx in range(precs.shape[1]):
        min_prec = 0
        for thr_idx in range(precs.shape[0]):
            if np.isnan(precs[thr_idx, cm_idx]):
                precs[thr_idx, cm_idx] = min_prec
            else:
                min_prec = precs[thr_idx, cm_idx]


    recalls = curves[:, 0, :] / (curves[:, 0, :] + curves[:, 3, :])
    f1s = 2*(precs*recalls)/(precs + recalls)
    accs = (curves[:, 0, :] + curves[:, 1, :]) / (curves.sum(axis=1))

    return precs, recalls, f1s, accs
    
