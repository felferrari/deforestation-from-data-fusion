from ops.ops import load_json, create_exps_paths, load_exp
import os
import multiprocessing
from train import train_model

exp = load_exp()
exp_n = exp['exp_n']
exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path = create_exps_paths(exp_n)


conf = load_json(os.path.join('conf', 'conf.json'))
n_train_models = conf['n_train_models']

if __name__ == '__main__':
    if os.path.exists(os.path.join(exp_path, 'model_losses.json')):
        os.remove(os.path.join(exp_path, 'model_losses.json'))
    model_idxs = [i for i in range(n_train_models)]
    for model_idx in model_idxs:
        p = multiprocessing.Process(target=train_model, args=(model_idx,))
        p.start()
        p.join()