from hp_motif import main

import pickle

import yaml
import numpy as np
from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.early_stop import no_progress_loss

import random
import torch

TASK = 'bbbp'

use_ln = [False, True]
activations = ['relu', 'softplus']

hp_space = {'threshold': hp.quniform('threshold', 0, 100, 10),
            'lr': hp.qloguniform('lr', -8, -6, 5e-4),
            'enc_dropout': hp.quniform('enc_dropout', 0.0, 0.5, 0.1),
            'tfm_dropout': hp.quniform('tfm_dropout', 0.0, 0.5, 0.1),
            'dec_dropout': hp.quniform('dec_dropout', 0.0, 0.5, 0.1),
            'enc_ln': hp.choice('enc_ln', use_ln), 
            'tfm_ln': hp.choice('tfm_ln', use_ln),
            'conc_ln': hp.choice('conc_ln', use_ln),
            'num_heads': hp.quniform('num_heads', 2, 4, 2)}


def objective(params):
    params['threshold'] = int(params['threshold'])
    params['lr'] = float(params['lr'])
    params['enc_dropout'] = float(params['enc_dropout'])
    params['tfm_dropout'] = float(params['tfm_dropout'])
    params['dec_dropout'] = float(params['dec_dropout'])
    params['enc_ln'] = use_ln[int(params['enc_ln'])]
    params['tfm_ln'] = use_ln[int(params['tfm_ln'])]
    params['conc_ln'] = use_ln[int(params['conc_ln'])]
    params['num_heads'] = int(params['num_heads'])

    print(params)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    res = []

    for __ in range(3):
        val_acc, __ = main(**params) 
        
        res.append(-val_acc)

    ret = sum(res) / len(res)
    print(ret)
    return ret

trials = Trials()

save_file_loc = 'dataset_' + TASK + '.pkl'

print(save_file_loc)

best = fmin(objective, hp_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=1000) 

print(best)
space_eval(hp_space, best)


pickle.dump(trials, open(save_file_loc, 'wb'))
