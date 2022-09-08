import logging
from ops.ops import load_opt_image, load_SAR_image, load_json, filter_outliers, load_label_image, load_SAR_DN_image
import fiona
import os
import numpy as np
from skimage.util import view_as_windows
import logging
import tqdm

def prep_tile(tile_id):

    conf = load_json(os.path.join('conf', 'conf.json'))
    patch_size = conf['patch_size']
    min_perc = conf['min_perc']
    n_opt_layers = conf['n_opt_layers']
    n_sar_layers = conf['n_sar_layers']

    train_step = int((1-conf['patch_overlap'])*patch_size)

    opt_path = os.path.join('img', 'GEE_imgs_opt')
    sar_path = os.path.join('img', 'GEE_imgs_sar')
    label_path = os.path.join('img', 'labels')
    cmap_scl_path = os.path.join('img', 'GEE_scl')
    patches_path = os.path.join('img', 'patches')

    logging.basicConfig(
        filename='02-prep.txt', 
        level=logging.INFO,
        filemode='a',
        format='%(asctime)s - %(message)s', 
        datefmt='%d-%b-%y %H:%M:%S'
        )

    logging.info(f'Preparing {tile_id} tile')

    opt = np.concatenate([
        filter_outliers(load_opt_image(os.path.join(opt_path, f'{tile_id}_opt_2019.tif'))), 
        filter_outliers(load_opt_image(os.path.join(opt_path, f'{tile_id}_opt_2020.tif')))
        ], axis=2)
    
    sar = np.concatenate([
        filter_outliers(load_SAR_image(os.path.join(sar_path, f'{tile_id}_sar_2019.tif'))), 
        filter_outliers(load_SAR_image(os.path.join(sar_path, f'{tile_id}_sar_2020.tif'))),
        ], axis=2)
    
    cmap_scl_19 = load_opt_image(os.path.join(cmap_scl_path, f'{tile_id}_cloud_scl_2019.tif'))
    cmap_scl_20 = load_opt_image(os.path.join(cmap_scl_path, f'{tile_id}_cloud_scl_2020.tif'))
    
    cmap = np.concatenate([
        np.expand_dims(cmap_scl_19[:,:,1], axis=-1),
        np.expand_dims(cmap_scl_20[:,:,1], axis=-1)
        ], axis=2)

    scl = np.concatenate([
        np.expand_dims(cmap_scl_19[:,:,0], axis=-1),
        np.expand_dims(cmap_scl_20[:,:,0], axis=-1)
        ], axis=2)

    del cmap_scl_19, cmap_scl_20

    label = np.expand_dims(load_label_image(os.path.join(label_path, f'label_{tile_id}.tif')), axis=-1)

    shape = label.shape[:2]

    idx_matrix = np.arange(shape[0]*shape[1]).reshape(shape)
    label_patches = view_as_windows(label, (patch_size, patch_size, 1), train_step).reshape((-1, patch_size, patch_size, 1))
    idx_patches = view_as_windows(idx_matrix, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))

    keep_patches = np.mean((label_patches == 1), axis=(1,2)).squeeze() >= min_perc
    idx_patches = idx_patches[keep_patches]

    np.savez(os.path.join(patches_path, f'{tile_id}.npz'),
        opt = opt.reshape((-1,n_opt_layers)),
        sar = sar.reshape((-1, n_sar_layers)),
        cmap = cmap.reshape((-1,2)),
        scl = scl.reshape((-1,2)),
        label = label.reshape((-1,1)),
        patches_idx = idx_patches,
        shape = shape
        )
    #np.save(os.path.join(patches_path, f'{tile_id}_opt.npy'), opt)
    #np.save(os.path.join(patches_path, f'{tile_id}_sar.npy'), sar)
    #np.save(os.path.join(patches_path, f'{tile_id}_cmap.npy'), cmap)
    #np.save(os.path.join(patches_path, f'{tile_id}_scl.npy'), scl)
    #np.save(os.path.join(patches_path, f'{tile_id}_label.npy'), label)
    #np.save(os.path.join(patches_path, f'{tile_id}_patches_idx.npy'), idx_patches)
    #np.save(os.path.join(patches_path, f'{tile_id}_shape.npy'), shape)

    logging.info(f'Tile {tile_id} produced {idx_patches.shape[0]} patches')

    return opt.mean(), opt.std(), sar.mean(), sar.std(), cmap.mean(), cmap.std()
