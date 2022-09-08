from ops.ops import load_json, create_exps_paths, load_exp
import os
import multiprocessing
from pred import predict
import fiona
import tqdm
import logging

if __name__ == '__main__':

    exp = load_exp()
    exp_n = exp['exp_n']

    exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path = create_exps_paths(exp_n)
    
    conf = load_json(os.path.join('conf', 'conf.json'))
    img_source = conf['img_source']

    paths = load_json(os.path.join('conf', 'paths.json'))
    shp_path = paths['shp']
    img_path = paths['img']

    shp = load_json(os.path.join('conf', 'shp.json'))
    grid_shp = shp[f'shp_download_{img_source}']
    grid_save = os.path.join(shp_path, f"{grid_shp}.shp")

    log_file = os.path.join(exp_path, f'pred.txt')
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO,
        filemode='a',
        format='%(asctime)s - %(message)s', 
        datefmt='%d-%b-%y %H:%M:%S'
        )

    #n_models = conf['n_models']

    test_feats = []
    with fiona.open(grid_save) as grid:
        for feat in grid:
            feat_id = int(feat['properties']['id'])
            #if feat['properties']['dataset'] == 0:
            test_feats.append(feat_id)
            

    for feat_id in tqdm.tqdm(test_feats):
        #predict(feat_id)
        p = multiprocessing.Process(target=predict, args=(feat_id,))
        p.start()
        p.join()