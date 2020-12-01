from data_manager import DataManager
from args import inputArguments
from args_2 import inputArguments_2
import numpy as np
import random
import torch

class makeDataset:
    def __init__(self):
        data_manager = DataManager(inputArguments())
        data_manager_2 = DataManager(inputArguments_2())

        train_activeset = data_manager.get_datasets()['train'] # type: <class 'dataset.TransformableFullMetadataSubset'>, len: 42236
        train_new_dataset = data_manager_2.get_datasets()['train']
        test_activeset = data_manager.get_datasets()['validation']
        test_new_dataset = data_manager_2.get_datasets()['validation']

        train_set = []
        for indx in range(len(train_activeset)):
            train_set.append(train_activeset[indx])

        known_indexes = []
        with open('kn_indici.txt', 'r') as file:
            for line in file:
                known_indexes.append(int(line.strip('\n')))

        known_set = []
        for indx in known_indexes:
            known_set.append(train_new_dataset[indx])

        test_indexes = []
        with open('test_indici_newdataset.txt', 'r') as file:
            for line in file:
                test_indexes.append(int(line.strip('\n')))

        test_set = []
        for indx in test_indexes:
            test_set.append(test_new_dataset[indx])

        grad_test_set = [] # solo per testare gradcam
        for indx in range(len(test_activeset)):
            test_set.append(test_activeset[indx])
            # Solo per testare gradcam
            grad_test_set.append(test_activeset[indx])


        expl_loader = torch.utils.data.DataLoader(grad_test_set,
                                                          batch_size=1,
                                                          shuffle=False, num_workers=2, pin_memory=True) 

        test_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=32,
                                                  shuffle=False, num_workers=2, pin_memory=True)

        self.tr = train_set
        self.ts = test_loader
        self.kn = known_set
        self.exp = expl_loader                                      

