import os
from pathlib import Path

import numpy as np
from dataset.data_module import EchoVideoDataModule
from models.model import EchoModel
from utils.train_utils import get_callbacks, get_weight_function
from utils.utils import fix_seed

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics import MeanSquaredError
from torchmetrics import MeanAbsoluteError
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import tensorboardX
import torch._dynamo as dynamo
dynamo.config.suppress_errors = True


def run_training(para_path=None,
                 train_path='data/merged_data_rvef/train_set_multilabel.json',
                 log_path='logs_regression_technical_full_model/'):
    '''Run training for the model
    
    Parameters
    ----------
    load_model : bool
        Whether to load a model from a checkpoint
    model_path : Path
        Path to the checkpoint
    base_path : str
        Path to the base directory for the experiment
    para_path : str
        Path to the yaml file containing the parameters
        
    Returns
    -------
    None
    '''
    with open(para_path, 'r') as f:
        parameters = yaml.safe_load(f)
    
    # Set seed
    fix_seed(parameters['seed'])

    augmentations = parameters['augmentations']

    parameters['train_path'] = train_path
    

    dataloader_params = parameters['dataloader']
    dataloader_params['to_predict'] = parameters['training']['to_predict']

    #train_path = 'data/merged_data_rvef/train_set_multilabel.json'
    val_path = 'data/rvenet_technical_lvef/validation_set_multilabel.json'
    test_path = 'data/rvenet_technical_lvef/test_set_multilabel.json'
    #test_path = None

    dm = EchoVideoDataModule(train_path, val_path, test_path=test_path,
                             augmentations=augmentations,
                             mae_config=parameters['model'],
                             **dataloader_params)
    
    weight_func = None
    if parameters['training']['weighted']:
        dm.setup()
        weight_func = get_weight_function(dm.train_data, parameters['training']['weight_alpha'])

    tensorboard_logger = pl_loggers.TensorBoardLogger(log_path)

    # Initialize model
    if 'post_pretraining_weights' in parameters['training']:
        ckpt_path = parameters['training']['post_pretraining_weights']
        model = EchoModel(parameters, weight_func=weight_func)
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        print('Loaded model from post-pretraining')
    else:
        model = EchoModel(parameters, weight_func=weight_func)

    #model = torch.compile(model)

    # Initialize callback
    callbacks = get_callbacks(parameters['callbacks'])

    # Initialize trainer
    trainer = pl.Trainer(accelerator='gpu',
                         devices=[1],
                         max_epochs=parameters['epochs'],
                         logger=tensorboard_logger,
                         callbacks=callbacks,
                         precision=parameters['precision'])

    # Train
    trainer.fit(model, dm)

    # Test
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer.test(model, datamodule=dm, ckpt_path=best_ckpt_path)


if __name__ == '__main__':
    freeze_params = 'data/rvenet_technical_lvef/parameters_freeze.yaml'
    nofreeze_params = 'data/rvenet_technical_lvef/parameters_no_freeze.yaml'
    nofreeze_params = 'data/rvenet_technical_lvef/parameters_shitty_pretrain.yaml'
    ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for ratio in ratios:
        log_path = f'logs_regression_technical/shitty_pretrain_{ratio}'
        train_path = f'data/rvenet_technical_lvef/train_set_multilabel_{ratio}.json'
        run_training(load_model=True, para_path=nofreeze_params, model_path='logs_pretraining/best_loss.ckpt', train_path=train_path, log_path=log_path)
    
    #for ratio in ratios:
    #    log_path = f'logs_regression_technical/freeze_{ratio}'
    #    train_path = f'data/rvenet_technical_lvef/train_set_multilabel_{ratio}.json'
    #    run_training(load_model=True, para_path=freeze_params, model_path='logs_pretraining/best_loss.ckpt', train_path=train_path, log_path=log_path)
