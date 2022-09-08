from tqdm.contrib.concurrent import process_map
import fiona
import os
from ops.ops import load_json
from prep import prep_tile
import numpy as np

if __name__ == '__main__':

    log_file = '02-prep.txt'
    if os.path.exists(log_file):
        os.remove(log_file)

    conf = load_json(os.path.join('conf', 'conf.json'))
    img_source = conf['img_source']

    shp = load_json(os.path.join('conf', 'shp.json'))
    grid_shp = shp[f'shp_download_{img_source}']

    grid_save = os.path.join('shp', f"{grid_shp}.shp")

    all_tiles = []
    with fiona.open(grid_save) as grid:
        for feat in grid:
            tile_id = int(feat['properties']['id'])
            all_tiles.append(tile_id)

    r = process_map(prep_tile, all_tiles, max_workers=8)
    #print(r)
    r = np.array(r).mean(axis=0)

    np.savez('statistics.npz',
        opt_mean = r[0],
        opt_std =  r[1],
        sar_mean =  r[2],
        sar_std =  r[3],
        cmap_mean =  r[4],
        cmap_std = r[5]
    )

