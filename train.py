import argparse
import numpy as np 
import pandas as pd
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tr
# from torchvision import models
from matplotlib import pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from time import time
import wandb
from create_model import Resnet34
import config_file
import data


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

"""
learning_rate:
beta1:
beta2:
batch_size:
weight_decay:
learning_rate_decay:
patience: 6

"""
parsar = argparse.ArgumentParser(description='demo argparse')
parsar.add_argument('-KP', '--KPath', required=True, type = str, help = 'K splits folder path')   ##not defaults
parsar.add_argument('-CP', '--CheckPointPath', required=True, type = str, help = 'Checkpoint path')
parsar.add_argument('-RN', '--run_name', required=True, type = str, help = 'run name')
parsar.add_argument('-E', '--epochs', required=True, type = int, help = 'Epochs') 
parsar.add_argument('-LR', '--lr', required=True, type = float, help = 'Learning Rate')    ########### R
parsar.add_argument('-B1', '--beta1', required=True, type = float, help = 'momentum beta1')  ############ R
parsar.add_argument('-B2', '--beta2', required=True, type = float, help = 'momentum beta2') ############ R
parsar.add_argument('-BS', '--batch_size', required=True, type = int, help = 'Batch Size')      ############# R
parsar.add_argument('-WD', '--weight_decay', required=True, type = float, help = 'Optimizer Weight Decay')   ############ R
parsar.add_argument('-LRS', '--lr_scheduler', type = str, help = 'learning rate scheduler', default='None')  ####### R
parsar.add_argument('-LRD', '--lr_decay', required=True, type = float, help = 'Learning Rate Decay')   #NR
parsar.add_argument('-P', '--Patience', type = int, help = 'Patience for RonPlateau', default = 6)   #NR

args = vars(parsar.parse_args())

wandb_config_defaults = {'lr': args['lr'],
                        'batch_size': args['batch_size'],
                        'beta1': args['beta1'],
                        'beta2': args['beta2'],
                        'weight_decay': args['weight_decay'],
                        'lr_scheduler': args['lr_scheduler'],
                        'epochs': args['epochs']
                        }

unrecorded_defaults = {'Patience': args['Patience'],
                 'lr_decay': args['lr_decay'],}

project_name = config_file.project_name
entity_name = config_file.entity_name
split_index = config_file.split_index

# print(args)
# print(project_name, entity_name)
# print(split_index)

if __name__=="__main__":
    split_index = config_file.split_index
    train_id_list, validation_id_list  = data.load_test_validation_df(args['KPath'], split_index)
    
    train_dataset = data.xray_dataset(train_id_list, expand = True, transforms = True, resize = 256)
    validation_datset = data.xray_dataset(validation_id_list, expand = True, transforms = False, resize = 256)
    #train_dataset = torch.utils.data.Subset(train_dataset, range(128)) ##########
    #validation_datset = torch.utils.data.Subset(validation_datset, range(128))  ############
    
    project_name = config_file.project_name
    entity_name = config_file.entity_name
    wandb.init(project = project_name, entity = entity_name, name = args['run_name'], config = wandb_config_defaults)
    config = wandb.config
    model = Resnet34(config, unrecorded_defaults)
    model.fit(train_dataset,
              validation_datset,
              epochs = config.epochs,
              batch_size = config.batch_size,
              num_workers = 2,
              checkpoint_metric = {'name':'accuracy', 'type': 'maximize'},
              wandb_p = wandb,
              model_checkpoint_path= args['CheckPointPath'],
              mixed_precision= True)
    
    






