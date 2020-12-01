import numpy as np
from os.path import join
from time import time
import sys
import math
import torch
import os
import argparse
import random

from resnet import resnet18
from data_manager import DataManager
from makeDS import makeDataset
import torch.nn.functional as F
from gradcam import GradCam

from torch.utils.tensorboard import SummaryWriter

max_iters = 2 # max iterations
epochs = 5 # total epochs for each iteration

# first strategy: random sampling
def _select_at_random(U):
    return np.random.choice(len(U))

# second strategy: uncertanty sampling
def _select_by_entropy(model, U):
        idxs_unlabeled = np.arange(len(U))
        probs = model.predict_proba(U)
        log_probs = torch.log(probs)
        E = (probs*log_probs).sum(1)
        return idxs_unlabeled[E.sort()[1][:1]]

# move an instance from src(unknown set) to dst(known set)
def _move(dst, src, i):  
    dst.append(src[i])
    src.pop(i)
    return dst, src

# active learning implementation
def _active_learning(experiment, model):
    tensorboard = True # Instantiate tensorboard writer
    if tensorboard:
        writer = SummaryWriter("./runs/")

    model.fit(experiment.kn,
              ite=0,
              max_ite=2,
              n_epochs=epochs,
              checkpoint=False)

    #grad_cam = GradCam(model.model, model.model.layer4, \
    #                       target_layer_names=["1"], use_cuda=True) # Instantiate gradcam

    # Learning loop begins
    for ite in range(max_iters):
        print('Lenght U:', len(experiment.tr))
        if not len(experiment.tr):
            break

        i = _select_by_entropy(model, experiment.tr) # or select_at_random
        experiment.kn, experiment.tr = _move(experiment.kn, experiment.tr, i) 

        if ite % 11 == 0:
            print('starting evaluation...')
            model.evaluate(experiment.ts, writer, ite)
            #print('Explaining....')
            #model.explain(grad_cam, experiment.exp, writer, ite)
       
        model.fit(experiment.kn,
                  ite,
                  max_iters,
                  n_epochs=epochs,
                  checkpoint=True)

    return 

if __name__ == '__main__':
    
    model = resnet18()
    experiment = makeDataset()
    _active_learning(experiment, model)