import numpy as np
import random
import tensorflow as tf
from ops.ops import load_json, patchs_full_gen
import os
import math as m

def train_data_gen(list_files, model_props):
    with np.load('statistics.npz', mmap_mode='r') as statistics:
        opt_mean = statistics['opt_mean']
        opt_std = statistics['opt_std']
        sar_mean = statistics['sar_mean']
        sar_std = statistics['sar_std']
        cmap_mean = statistics['cmap_mean']
        cmap_std = statistics['cmap_std']
    patches_path = os.path.join('img', 'patches')
    model_input = model_props['input']
    model_output = model_props['output']
    def func():
        while True:
            #random.shuffle(list_files)
            for fin in list_files:
                with np.load(os.path.join(patches_path, f'{fin}.npz'), mmap_mode='r') as data:
                    #print(fin)
                    patches_idx = data['patches_idx']
                    
                    if model_input == 'opt':
                        input = data['opt']
                    if model_input == 'sar':
                        input = data['sar']
                    if model_input in {'opt+sar', 'opt_sar'}:
                        input_opt = data['opt']
                        input_sar = data['sar']
                    if model_input in {'opt_sar_cmap', 'opt+sar_sar_cmap', 'opt_sar_sar_cmap'}:
                        input_opt = data['opt']
                        input_sar = data['sar']
                        input_cmap = data['cmap']

                    if model_output == 'pred_fus':
                        label = data['label']

                for patch in patches_idx:
                    if model_output == 'pred_fus':
                        if model_input in {'opt', 'sar'}:
                            yield input[patch].astype(np.float32), label[patch].astype(np.uint8)
                        if model_input in {'opt+sar', 'opt_sar'}:
                            yield input_opt[patch].astype(np.float32), input_sar[patch].astype(np.float32), label[patch].astype(np.uint8)
                        if model_input in {'opt_sar_cmap', 'opt+sar_sar_cmap', 'opt_sar_sar_cmap'}:
                            yield input_opt[patch].astype(np.float32), input_sar[patch].astype(np.float32), input_cmap[patch].astype(np.float32), label[patch].astype(np.uint8)

    return func

class PredictDataGen(tf.keras.utils.Sequence):
    def __init__(self, fin, model_props, batch_size):

        patches_path = os.path.join('img', 'patches')

        self.batch_size = batch_size
        self.model_input = model_props['input']
        self.model_output = model_props['output']
        self.counter = 0

        conf = load_json(os.path.join('conf', 'conf.json'))
        patch_size = conf['patch_size']
        test_patch_step = conf['test_patch_step']

        with np.load('statistics.npz', mmap_mode='r') as statistics:
            opt_mean = statistics['opt_mean']
            opt_std = statistics['opt_std']
            sar_mean = statistics['sar_mean']
            sar_std = statistics['sar_std']
            cmap_mean = statistics['cmap_mean']
            cmap_std = statistics['cmap_std']

        with np.load(os.path.join(patches_path, f'{fin}.npz'), mmap_mode='r') as data:
            self.img_shape = data['shape']
            idx_matrix = np.arange(self.img_shape[0]*self.img_shape[1]).reshape(self.img_shape)
            self.patches_idx = patchs_full_gen(idx_matrix, patch_size, test_patch_step)
            self.shape_pred_patches = self.patches_idx.shape[:2]
            self.patches_idx = self.patches_idx.reshape((-1, patch_size, patch_size))

            if self.model_input == 'opt':
                self.data = (data['opt'] - opt_mean) / opt_std
            if self.model_input == 'sar':
                self.data = (data['sar'] - sar_mean) / sar_std
            if self.model_input == 'opt+sar':
                x_opt = (data['opt'] - opt_mean) / opt_std
                x_sar = (data['sar'] - sar_mean) / sar_std
                self.data = np.concatenate([x_opt, x_sar], axis=-1)
            if self.model_input == 'opt_sar':
                self.x_opt = (data['opt'] - opt_mean) / opt_std
                self.x_sar = (data['sar'] - sar_mean) / sar_std
            if self.model_input in {'opt_sar_cmap', 'opt+sar_sar_cmap'}:
                x_opt = (data['opt'] - opt_mean) / opt_std
                self.x_sar = (data['sar'] - sar_mean) / sar_std
                self.x_fus1 = np.concatenate([x_opt, self.x_sar], axis=-1)
                self.x_cmap = (data['cmap'] - cmap_mean) / cmap_std
            if self.model_input in {'opt_sar_sar_cmap'}:
                self.x_opt = (data['opt'] - opt_mean) / opt_std
                self.x_sar = (data['sar'] - sar_mean) / sar_std
                self.x_cmap = (data['cmap'] - cmap_mean) / cmap_std
  
           

    def __len__(self):
        return m.ceil((self.shape_pred_patches[0]*self.shape_pred_patches[1])/self.batch_size)

    def __getitem__(self, index):
        batch_patches_idx = self.patches_idx[index*self.batch_size: (index+1)*self.batch_size]
        if self.model_input in {'opt', 'sar', 'opt+sar'}:
            return tf.convert_to_tensor(self.data[batch_patches_idx])
        if self.model_input == 'opt_sar':
            return tf.convert_to_tensor(self.x_opt[batch_patches_idx]), tf.convert_to_tensor(self.x_sar[batch_patches_idx])
        if self.model_input == 'opt_sar_cmap':
            return tf.convert_to_tensor(self.x_opt[batch_patches_idx]), tf.convert_to_tensor(self.x_sar[batch_patches_idx]), tf.convert_to_tensor(self.x_cmap[batch_patches_idx])
        if self.model_input == 'opt+sar_sar_cmap':
            return tf.convert_to_tensor(self.x_fus1[batch_patches_idx]), tf.convert_to_tensor(self.x_sar[batch_patches_idx]), tf.convert_to_tensor(self.x_cmap[batch_patches_idx])
        if self.model_input == 'opt_sar_sar_cmap':
            return (
                (tf.convert_to_tensor(self.x_opt[batch_patches_idx]), tf.convert_to_tensor(self.x_sar[batch_patches_idx])),
                tf.convert_to_tensor(self.x_sar[batch_patches_idx]),
                tf.convert_to_tensor(self.x_cmap[batch_patches_idx])
            )
            
       

    def __iter__(self):
        return self
    
    def __call__(self):
        return self.next()
    
    def __next__(self):
        self.counter+=1
        if self.counter > len(self):
            raise StopIteration
        else:
            return self[self.counter-1]

def prep_data(model_props, normalize = True):
    conf = load_json(os.path.join('conf', 'conf.json'))
    n_opt_layers = conf['n_opt_layers']
    n_sar_layers = conf['n_sar_layers']
    n_cmap_layers = conf['n_cmap_layers']
    n_classes = conf['n_classes']
    model_input = model_props['input']
    model_output = model_props['output']
    if normalize:
        with np.load('statistics.npz', mmap_mode='r') as statistics:
            opt_mean = statistics['opt_mean']
            opt_std = statistics['opt_std']
            sar_mean = statistics['sar_mean']
            sar_std = statistics['sar_std']
            cmap_mean = statistics['cmap_mean']
            cmap_std = statistics['cmap_std']
    def prep_func(*data):
        #prepare data
        if model_output == 'pred_fus':
            y = tf.one_hot(tf.squeeze(data[-1]), n_classes)

            if model_input == 'opt':
                x = (data[0] - opt_mean) / opt_std if normalize else data[0]
                return x, y
            if model_input == 'sar':
                x = (data[0] - sar_mean) / sar_std if normalize else data[0]
                return x, y
            if model_input == 'opt+sar':
                x_opt = (data[0] - opt_mean) / opt_std if normalize else data[0]
                x_sar = (data[1] - sar_mean) / sar_std if normalize else data[1]
                x = tf.concat([x_opt, x_sar], axis=-1)
                return x, y
            if model_input == 'opt_sar':
                x_opt = (data[0] - opt_mean) / opt_std if normalize else data[0]
                x_sar = (data[1] - sar_mean) / sar_std if normalize else data[1]
                return (x_opt, x_sar), y
            if model_input == 'opt_sar_cmap':
                x_opt = (data[0] - opt_mean) / opt_std if normalize else data[0]
                x_sar = (data[1] - sar_mean) / sar_std if normalize else data[1]
                x_cmap = (data[2] - cmap_mean) / cmap_std if normalize else data[2]
                return (x_opt, x_sar, x_cmap), y

            if model_input == 'opt+sar_sar_cmap':
                x_opt = (data[0] - opt_mean) / opt_std if normalize else data[0]
                x_sar = (data[1] - sar_mean) / sar_std if normalize else data[1]
                x_cmap = (data[2] - cmap_mean) / cmap_std if normalize else data[2]
                return (tf.concat([x_opt, x_sar], axis=-1), x_sar, x_cmap), y

            if model_input == 'opt_sar_sar_cmap':
                x_opt = (data[0] - opt_mean) / opt_std if normalize else data[0]
                x_sar = (data[1] - sar_mean) / sar_std if normalize else data[1]
                x_cmap = (data[2] - cmap_mean) / cmap_std if normalize else data[2]
                return ((x_opt, x_sar), x_sar, x_cmap), y

    return prep_func        

def data_augmentation(model_props):
    model_input = model_props['input']
    model_output = model_props['output']
    def da_fn(*data):
        #print('data aug')
        #print(data)
        if model_output == 'pred_fus':
            y = data[-1]
            if model_input in {'opt', 'sar', 'opt+sar'}:
                x = data[0]
                if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
                    x = tf.image.flip_left_right(x)
                    y = tf.image.flip_left_right(y)
                if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
                    x = tf.image.flip_up_down(x)
                    y = tf.image.flip_up_down(y)

                k = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
                x = tf.image.rot90(x, k)
                y = tf.image.rot90(y, k)
            elif model_input in {'opt_sar'}:
                x_opt = data[0][0]
                x_sar = data[0][1]
                if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
                    x_opt = tf.image.flip_left_right(x_opt)
                    x_sar = tf.image.flip_left_right(x_sar)
                    y = tf.image.flip_left_right(y)
                if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
                    x_opt = tf.image.flip_up_down(x_opt)
                    x_sar = tf.image.flip_up_down(x_sar)
                    y = tf.image.flip_up_down(y)

                k = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
                x_opt = tf.image.rot90(x_opt, k)
                x_sar = tf.image.rot90(x_sar, k)
                y = tf.image.rot90(y, k)
                x = (x_opt, x_sar)
            elif model_input in {'opt_sar_cmap', 'opt+sar_sar_cmap'}:
                x_opt = data[0][0]
                x_sar = data[0][1]
                x_cmap = data[0][2]
                if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
                    x_opt = tf.image.flip_left_right(x_opt)
                    x_sar = tf.image.flip_left_right(x_sar)
                    x_cmap = tf.image.flip_left_right(x_cmap)
                    y = tf.image.flip_left_right(y)
                if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
                    x_opt = tf.image.flip_up_down(x_opt)
                    x_sar = tf.image.flip_up_down(x_sar)
                    x_cmap = tf.image.flip_up_down(x_cmap)
                    y = tf.image.flip_up_down(y)

                k = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
                x_opt = tf.image.rot90(x_opt, k)
                x_sar = tf.image.rot90(x_sar, k)
                x_cmap = tf.image.rot90(x_cmap, k)
                y = tf.image.rot90(y, k)
                x = (x_opt, x_sar, x_cmap)

            elif model_input == 'opt_sar_sar_cmap':
                x_opt = data[0][0][0]
                x_sar = data[0][0][1]
                x_cmap = data[0][2]
                if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
                    x_opt = tf.image.flip_left_right(x_opt)
                    x_sar = tf.image.flip_left_right(x_sar)
                    x_cmap = tf.image.flip_left_right(x_cmap)
                    y = tf.image.flip_left_right(y)
                if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
                    x_opt = tf.image.flip_up_down(x_opt)
                    x_sar = tf.image.flip_up_down(x_sar)
                    x_cmap = tf.image.flip_up_down(x_cmap)
                    y = tf.image.flip_up_down(y)

                k = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
                x_opt = tf.image.rot90(x_opt, k)
                x_sar = tf.image.rot90(x_sar, k)
                x_cmap = tf.image.rot90(x_cmap, k)
                y = tf.image.rot90(y, k)
                x = ((x_opt, x_sar), x_sar, x_cmap)
                
            return x, y
       
    return da_fn

def get_dataset(list_files, model_props):
    conf = load_json(os.path.join('conf', 'conf.json'))
    patch_size = conf['patch_size']
    batch_size = conf['batch_size']
    n_opt_layers = conf['n_opt_layers']
    n_sar_layers = conf['n_sar_layers']
    n_cmap_layers = conf['n_cmap_layers']
    model_input = model_props['input']
    model_output = model_props['output']

    if model_output == 'pred_fus':
        if model_input == 'opt':
            output_signature = (
                tf.TensorSpec(shape=(patch_size , patch_size , n_opt_layers), dtype=tf.float32),
                tf.TensorSpec(shape=(patch_size , patch_size , 1), dtype=tf.uint8),
            )
        if model_input == 'sar':
            output_signature = (
                tf.TensorSpec(shape=(patch_size , patch_size , n_sar_layers), dtype=tf.float32),
                tf.TensorSpec(shape=(patch_size , patch_size , 1), dtype=tf.uint8),
            )
        if model_input in {'opt+sar', 'opt_sar'}:
            output_signature = (
                tf.TensorSpec(shape=(patch_size , patch_size , n_opt_layers), dtype=tf.float32),
                tf.TensorSpec(shape=(patch_size , patch_size , n_sar_layers), dtype=tf.float32),
                tf.TensorSpec(shape=(patch_size , patch_size , 1), dtype=tf.uint8),
            )
        if model_input in {'opt_sar_cmap', 'opt+sar_sar_cmap', 'opt_sar_sar_cmap'}:
            output_signature = (
                tf.TensorSpec(shape=(patch_size , patch_size , n_opt_layers), dtype=tf.float32),
                tf.TensorSpec(shape=(patch_size , patch_size , n_sar_layers), dtype=tf.float32),
                tf.TensorSpec(shape=(patch_size , patch_size , n_cmap_layers), dtype=tf.float32),
                tf.TensorSpec(shape=(patch_size , patch_size , 1), dtype=tf.uint8),
            )
        
    
    
    ds = tf.data.Dataset.from_generator(
        generator = train_data_gen(list_files, model_props), 
        output_signature = output_signature
        )
    return ds

def get_n_steps(list_files):
    n = 0
    patches_path = os.path.join('img', 'patches')
    for fin in list_files:
        data = np.load(os.path.join(patches_path, f'{fin}.npz'), mmap_mode='r')
        n += data['patches_idx'].shape[0]
    return n

    