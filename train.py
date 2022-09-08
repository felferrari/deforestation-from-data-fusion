from ops.ops import load_exp, create_exps_paths, load_json, save_json
import os
import logging
import fiona
from dataset import get_dataset, data_augmentation, get_n_steps, prep_data
from models.model import resunet, resunet_base, early_fusion_resunet, late_fusion_resunet
from models import model as model_module
from models.callbacks import ResetGenerator
import tensorflow as tf
from models.losses import WBCE
import time
import math as m
import numpy as np


def get_model(model_props):
    conf = load_json(os.path.join('conf', 'conf.json'))
    model_size = conf['model_size']
    patch_size = conf['patch_size']
    n_classes = conf['n_classes']
    reg_weight = conf['reg_weight']

    model_fn = model_props['function']
    model_input = model_props['input']
    model_name = model_props['name']
    #model_output = model_props['output']

    if model_fn == 'resunet':
        if model_input == 'opt':
            shape_in = (patch_size, patch_size, conf['n_opt_layers'])
        elif model_input == 'sar':
            shape_in = (patch_size, patch_size, conf['n_sar_layers'])
        elif model_input == 'opt+sar':
            shape_in = (patch_size, patch_size, conf['n_opt_layers'] + conf['n_sar_layers'])

        model = resunet(shape_in, model_size, n_classes, reg_weight, name = model_name)
    if model_fn == 'early_fusion_resunet':
        opt_shape_in = (patch_size, patch_size, conf['n_opt_layers'])
        sar_shape_in = (patch_size, patch_size, conf['n_sar_layers'])
        model = early_fusion_resunet(opt_shape_in, sar_shape_in, model_size, n_classes, reg_weight, name = model_name)
    if model_fn  == 'late_fusion_resunet':
        opt_shape_in = (patch_size, patch_size, conf['n_opt_layers'])
        sar_shape_in = (patch_size, patch_size, conf['n_sar_layers'])
        model = late_fusion_resunet(opt_shape_in, sar_shape_in, model_size, n_classes, reg_weight, name = model_name)

 

    return model

def compile_model(model, model_props):
    conf = load_json(os.path.join('conf', 'conf.json'))
    learning_rate = conf['learning_rate']
    class_weights = conf['class_weights']
    run_eagerly = conf['run_eagerly']

    model_output = model_props['output']

    #optimizer = tf.keras.optimizers.SGD(learning_rate)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.999
        )

    if model_output == 'pred_fus':
        loss = WBCE(class_weights)
        #loss = tf.keras.losses.BinaryCrossentropy()
        metrics = ['accuracy']

        model.compile(
            loss=loss,
            optimizer = optimizer,
            run_eagerly=run_eagerly,
            metrics = metrics
        )

    '''elif model_name == 'multi_fus':
        metrics = {
            'optical':'accuracy',
            'fusion':'accuracy',
            'cloud':'mse'
        }
        loss = {
            'optical':WBCE(class_weights),
            'fusion':WBCE(class_weights),
            'cloud':'mse'
        }
        lossWeights = {
            'optical':conf['opt_weight'],
            'fusion':conf['fus_weight'],
            'cloud':conf['cloud_weight'],
        }
        model.compile(
            loss=loss,
            loss_weights = lossWeights,
            optimizer = optimizer,
            run_eagerly=run_eagerly,
            metrics = metrics
        )'''

def get_callbacks(model_idx):
    conf = load_json(os.path.join('conf', 'conf.json'))
    train_patience = conf['train_patience']
    min_lr = conf['min_lr']

    exp = load_exp()
    exp_n = exp['exp_n']
    exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path = create_exps_paths(exp_n)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        restore_best_weights = True,
        patience=train_patience
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(logs_path, f'log_tb_{model_idx}'),
        histogram_freq = 10
        )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(models_path, f'model_{model_idx}'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        save_freq='epoch',
    )
    return [
        #lr_control,
        #model_checkpoint,
        ResetGenerator(),
        tensorboard,
        early_stop
    ]


def prepare_datasets(model_props, train_l, val_l, test_l, batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = get_dataset(train_l, model_props)
    train_ds = train_ds.map(prep_data(model_props), num_parallel_calls=AUTOTUNE, deterministic = False)
    train_ds = train_ds.map(data_augmentation(model_props), num_parallel_calls=AUTOTUNE, deterministic = False)
    train_ds = train_ds.shuffle(50*batch_size)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = get_dataset(val_l, model_props)
    val_ds = val_ds.map(prep_data(model_props), num_parallel_calls=AUTOTUNE, deterministic = False)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(AUTOTUNE)

    test_ds = get_dataset(test_l, model_props)
    test_ds = test_ds.map(prep_data(model_props), num_parallel_calls=AUTOTUNE, deterministic = False)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds

def train_model(train_idx):
    
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    t0 = time.perf_counter()

    exp = load_exp()
    exp_n = exp['exp_n']
    model_props = exp['model_props']
    model_name = model_props['name']
    exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path = create_exps_paths(exp_n)

    conf = load_json(os.path.join('conf', 'conf.json'))
    img_source = conf['img_source']
    max_epochs = conf['max_epochs']
    batch_size = conf['batch_size']
    learning_rate = conf['learning_rate']
    n_train_models = conf['n_train_models']
    model_size = conf['model_size']
    patch_size = conf['patch_size']
    n_classes = conf['n_classes']
    reg_weight = conf['reg_weight']

    paths = load_json(os.path.join('conf', 'paths.json'))
    shp_path = paths['shp']
    img_path = paths['img']

    shp = load_json(os.path.join('conf', 'shp.json'))
    grid_shp = shp[f'shp_download_{img_source}']
    grid_save = os.path.join(shp_path, f"{grid_shp}.shp")

    patches_path = os.path.join(img_path, paths['patches'])

    log_file = os.path.join(exp_path, f'train_{train_idx}.txt')
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO,
        filemode='a',
        format='%(asctime)s - %(message)s', 
        datefmt='%d-%b-%y %H:%M:%S'
        )

    train_l = []
    val_l = []
    test_l = []
    with fiona.open(grid_save) as grid:
        for feat in grid:
            feat_id = int(feat['properties']['id'])
            if feat['properties']['dataset'] == 0:
                train_l.append(feat_id)
            elif feat['properties']['dataset'] == 1:
                val_l.append(feat_id)
            elif feat['properties']['dataset'] == 2:
                test_l.append(feat_id)

    train_steps = m.ceil(get_n_steps(train_l)/batch_size)
    val_steps = m.ceil(get_n_steps(val_l)/batch_size)
    test_steps = m.ceil(get_n_steps(test_l)/batch_size)

    if not model_props['combined_model']:

        train_ds, val_ds, test_ds = prepare_datasets(model_props, train_l, val_l, test_l, batch_size)

        min_val_loss = {}

        print(f'Training {model_props["name"]} model {train_idx+1} / {n_train_models}')
        model = get_model(model_props)
        compile_model(model, model_props)
        callbacks = get_callbacks(train_idx)
        
        history = model.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs = max_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            verbose=2,
            callbacks = callbacks
        )

        val_loss = np.array(history.history['val_loss']).min()

        model.save(
            os.path.join(models_path, f'model_{train_idx}'),
            options=tf.saved_model.SaveOptions(save_debug_info=True)
            )

        model.evaluate(
            x=test_ds,
            steps=test_steps
        )

        model.summary(
            print_fn = logging.info
        )
        if os.path.exists(os.path.join(exp_path, 'model_losses.json')):
            min_val_loss = load_json(os.path.join(exp_path, 'model_losses.json'))
        else:
            min_val_loss = {}
        min_val_loss[f'model_{train_idx}'] = val_loss
        save_json(min_val_loss, os.path.join(exp_path, 'model_losses.json'))

        logging.info(f'Train consumed {(time.perf_counter() - t0)/60} mins')

    else: #if combined_model == True

        model_1_props = load_exp(model_props['model_1'])['model_props']
        model_1_name = model_1_props['name']

        train_ds, val_ds, test_ds = prepare_datasets(model_1_props, train_l, val_l, test_l, batch_size)

        print(f'Training {model_1_name} model {train_idx+1} / {n_train_models}')
        model_1 = get_model(model_1_props)
        compile_model(model_1, model_1_props)
        callbacks = get_callbacks(train_idx)
        
        history = model_1.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs = max_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            verbose=2,
            callbacks = callbacks
        )

        model_1.summary(
            print_fn = logging.info
        )

        
        model_2_props = load_exp(model_props['model_2'])['model_props']
        model_2_name = model_2_props['name']

        train_ds, val_ds, test_ds = prepare_datasets(model_2_props, train_l, val_l, test_l, batch_size)

        print(f'Training {model_2_name} model {train_idx+1} / {n_train_models}')
        model_2 = get_model(model_2_props)
        compile_model(model_2, model_2_props)
        callbacks = get_callbacks(train_idx)
        
        history = model_2.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs = max_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            verbose=2,
            callbacks = callbacks
        )

        model_2.summary(
            print_fn = logging.info
        )

        shape_cmap_in = (patch_size, patch_size, conf['n_cmap_layers'])


        model_fn = getattr(model_module, model_props['function'])
        

        train_ds, val_ds, test_ds = prepare_datasets(model_props, train_l, val_l, test_l, batch_size)

        min_val_loss = {}

        print(f'Training {model_props["name"]} model {train_idx+1} / {n_train_models}')
        #model = get_model(model_props)
        model_fus = model_fn(model_1, model_2, shape_cmap_in, model_size, n_classes, reg_weight, name = model_name)
        compile_model(model_fus, model_props)
        callbacks = get_callbacks(train_idx)
        
        history = model_fus.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs = max_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            verbose=2,
            callbacks = callbacks
        )

        val_loss = np.array(history.history['val_loss']).min()

        model_fus.save(
            os.path.join(models_path, f'model_{train_idx}'),
            options=tf.saved_model.SaveOptions(save_debug_info=True)
            )

        model_fus.evaluate(
            x=test_ds,
            steps=test_steps
        )

        model_fus.summary(
            print_fn = logging.info
        )
        if os.path.exists(os.path.join(exp_path, 'model_losses.json')):
            min_val_loss = load_json(os.path.join(exp_path, 'model_losses.json'))
        else:
            min_val_loss = {}
        min_val_loss[f'model_{train_idx}'] = val_loss
        save_json(min_val_loss, os.path.join(exp_path, 'model_losses.json'))

        logging.info(f'Train consumed {(time.perf_counter() - t0)/60} mins')






        '''model_opt.trainable = False
        model_sar.trainable = False
        cmap_input = tf.keras.layers.Input(shape_cmap_in, name = 'cmap_input')
        concat_fus = tf.keras.layers.Concatenate(name='fus_concat_0')([model_opt.layers[-3].output, model_sar.layers[-3].output, cmap_input])
        out_fus = resunet_base(concat_fus, model_size, n_classes, reg_weight, name = 'fus')
        
        model_fus = tf.keras.Model(inputs = [model_opt.input, model_sar.input, cmap_input], outputs = out_fus)

        model_name = exp['model_name']

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_ds = get_dataset(train_l, model_props)
        train_ds = train_ds.map(prep_data(model_props), num_parallel_calls=AUTOTUNE, deterministic = False)
        train_ds = train_ds.map(data_augmentation(model_props), num_parallel_calls=AUTOTUNE, deterministic = False)
        train_ds = train_ds.shuffle(50*batch_size)
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(AUTOTUNE)

        val_ds = get_dataset(val_l, model_props)
        val_ds = val_ds.map(prep_data(model_props), num_parallel_calls=AUTOTUNE, deterministic = False)
        val_ds = val_ds.batch(batch_size)
        val_ds = val_ds.prefetch(AUTOTUNE)

        test_ds = get_dataset(test_l, model_props)
        test_ds = test_ds.map(prep_data(model_props), num_parallel_calls=AUTOTUNE, deterministic = False)
        test_ds = test_ds.batch(batch_size)
        test_ds = test_ds.prefetch(AUTOTUNE)

        train_steps = m.ceil(get_n_steps(train_l)/batch_size)
        val_steps = m.ceil(get_n_steps(val_l)/batch_size)
        test_steps = m.ceil(get_n_steps(test_l)/batch_size)

        print(f'Training fusion model {train_idx+1} / {n_train_models}')
        compile_model(model_fus, model_props)
        callbacks = get_callbacks(train_idx)
        
        history = model_fus.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs = max_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            verbose=2,
            callbacks = callbacks
        )


        val_loss = np.array(history.history['val_loss']).min()

        model_fus.save(
            os.path.join(models_path, f'model_{train_idx}'),
            options=tf.saved_model.SaveOptions(save_debug_info=True)
            )

        model_fus.evaluate(
            x=test_ds,
            steps=test_steps
        )

        model_fus.summary(
            print_fn = logging.info
        )
        if os.path.exists(os.path.join(exp_path, 'model_losses.json')):
            min_val_loss = load_json(os.path.join(exp_path, 'model_losses.json'))
        else:
            min_val_loss = {}
        min_val_loss[f'model_{train_idx}'] = val_loss
        save_json(min_val_loss, os.path.join(exp_path, 'model_losses.json'))

        logging.info(f'Train consumed {(time.perf_counter() - t0)/60} mins')
'''


    return
