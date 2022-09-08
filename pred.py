from ops.ops import load_exp, create_exps_paths, load_json, rebuild_from_patches
import os
import logging
from dataset import PredictDataGen, get_dataset
from skimage.util import crop
import numpy as np
from train import get_model, compile_model
import tensorflow as tf
import time
import operator

def predict(feat_id):
    t0 = time.perf_counter()
    exp = load_exp()
    exp_n = exp['exp_n']
    model_props = exp['model_props']
    exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path = create_exps_paths(exp_n)

    conf = load_json(os.path.join('conf', 'conf.json'))
    img_source = conf['img_source']
    batch_size = conf['batch_size']
    patch_size = conf['patch_size']
    test_patch_step = conf['test_patch_step']
    n_classes = conf['n_classes']
    n_inference_models = conf['n_inference_models']

    paths = load_json(os.path.join('conf', 'paths.json'))
    img_path = paths['img']


    patches_path = os.path.join(img_path, paths['patches'])

    crop_l = int((patch_size - test_patch_step)/2)
    crop_w = (
                (0,0),
                (crop_l,crop_l),
                (crop_l,crop_l),
                (0,0)
            )

    #test_in = os.path.join(patches_path, f'{feat_id}.npz')
    pred_ds = PredictDataGen(feat_id, model_props, batch_size)
    pred_images = np.zeros((n_inference_models,pred_ds.img_shape[0], pred_ds.img_shape[1], n_classes))
    
    models_losses = load_json(os.path.join(exp_path, 'model_losses.json'))
    models_idx = sorted(models_losses.items(), key=operator.itemgetter(1))

    for i, model_idx in enumerate(models_idx[:n_inference_models]):
        print(f'Predicting {model_idx[0]}')
        model = tf.keras.models.load_model(os.path.join(models_path, model_idx[0]), compile=False )
        compile_model(model, model_props)
        #pred_ds = PredictDataGen(feat_id, model_name, batch_size)

        pred = None
        for batch_i in range(len(pred_ds)):
            batch_data = pred_ds[batch_i]
            pred_batch = model.predict_on_batch(
                batch_data
                )#.reshape(pred_ds.shape_pred_patches+(patch_size, patch_size, n_classes))
            if pred is None:
                pred = pred_batch
            else:
                pred = np.concatenate([pred, pred_batch], axis=0)
        pred = pred.reshape(pred_ds.shape_pred_patches+(patch_size, patch_size, n_classes))
        predicted  = None
        for pred in pred:
            if predicted is None:
                predicted = crop(pred, crop_w)
            else:
                predicted = np.concatenate([predicted, crop(pred, crop_w)], axis=0)

        predicted = predicted.reshape((pred_ds.shape_pred_patches)+(test_patch_step, test_patch_step, n_classes))
        pred_images[i] = rebuild_from_patches(predicted, pred_ds.img_shape)

    mean_pred = pred_images.mean(axis=0)
    np.save(os.path.join(predictions_path, f'{feat_id}_fus.npy'), mean_pred)
    logging.info(f'Prediction of tile {feat_id} consumed {(time.perf_counter() - t0)/60} mins')