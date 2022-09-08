import fiona
import numpy as np
import os
from ops.ops import load_json, create_exps_paths, load_exp
from eval import plot_sample, eval_feat_curves, plot_ap_curves, eval_metrics_clouds, eval_feat, plot_metrics_curves
import matplotlib.pyplot as plt
from multiprocessing import Pool
import tqdm
import tensorflow as tf
import pandas as pd

conf = load_json(os.path.join('conf', 'conf.json'))
exp = load_exp()
paths = load_json(os.path.join('conf', 'paths.json'))
shp = load_json(os.path.join('conf', 'shp.json'))
limits = load_json(os.path.join('conf', 'limits.json'))
conf = load_json(os.path.join('conf', 'conf.json'))
shp_path = paths['shp']

exp_n = exp['exp_n']
exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path = create_exps_paths(exp_n)

log_file = os.path.join(exp_path, f'eval.txt')

img_source = conf['img_source']
grid_shp = shp[f'shp_download_{img_source}']
grid_save = os.path.join('shp', f"{grid_shp}.shp")

train_patches_path = os.path.join('img', 'patches')

test_patch_step = conf['test_patch_step']       
n_classes = conf['n_classes']
model_size = conf['model_size']
class_weights = conf['class_weights']
patch_size = conf['patch_size']
n_cmap_layers = conf['n_cmap_layers']

cmap_lim = conf['cmap_lim']
n_cmap_lim = len(cmap_lim) + 1

test_feat_ids = []
all_feats = []
with fiona.open(grid_save) as grid:
    for feat in grid:
        feat_id = int(feat['properties']['id'])
        
        all_feats.append(feat_id)
        if feat['properties']['dataset'] == 2: # test dataset
            test_feat_ids.append(feat_id)


if __name__ == '__main__':
    
    for feat_id in tqdm.tqdm(all_feats):

        break
        
        fz = np.load(os.path.join(train_patches_path, f'{feat_id}.npz'))
        shape = fz['shape']
        shape = (shape[0], shape[1])
        label = tf.keras.utils.to_categorical(fz['label'], conf['n_classes']).reshape(shape+(n_classes,))
        pred = np.load(os.path.join(predictions_path, f'{feat_id}_fus.npy')).reshape(shape+(n_classes,))[:,:,1]
        cmap = fz['cmap'].reshape(shape+(n_cmap_layers,))

        plot_sample(pred, cmap, label, os.path.join(visual_path, f'result_{feat_id}.png'))
    
    if os.path.exists(log_file):
        os.remove(log_file)

    def_val = eval_feat(6.25)
    precision = def_val[ 0, :] / (def_val[0,:] + def_val[2,:])
    recall = def_val[ 0, :] / (def_val[0,:] + def_val[3,:])
    f1 = (2 * precision * recall)/(precision + recall)

    pd.DataFrame({
        'Cloud Cond': ['No Cloud', 'Medium Cloud', 'High Cloud', 'Global'],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }).to_excel(os.path.join(visual_path, f'metrics.xlsx'))

    with Pool(processes=8) as p:
        limits = np.linspace(0, 50, 51) #0, 100, 51
        curves = p.map(eval_feat, limits)
        curves = np.array(curves)

        plot_metrics_curves(curves, limits, os.path.join(visual_path, f'precision.png'))



    '''precs, recalls, f1scores, accs = eval_metrics_clouds(np.array(curves))
    precs = np.flip(precs, axis=0)
    recalls = np.flip(recalls, axis=0)
    d_rec = recalls[1:] - recalls[:-1]
    aps = (d_rec * precs[1:]).sum(axis=0)
    plot_ap_curves(precs, recalls, aps, os.path.join(visual_path, f'ap_curves.png'))
'''
    
            