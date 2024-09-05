import os
from pathlib import Path
from argparse import ArgumentParser

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
                 data_path=None,
                 log_path='logs/'):
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

    dataloader_params = parameters['dataloader']
    dataloader_params['to_predict'] = parameters['training']['to_predict']

    data_path = Path(data_path)

    train_path = list(data_path.glob('train_set*.json'))[0]
    val_path = list(data_path.glob('validation_set*.json'))[0]
    test_path = list(data_path.glob('test_set*.json'))[0] if 'test_set' in [x.stem for x in data_path.glob('test_set*.json')] else None
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


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--log_path', type=str, default='logs/')

    args = parser.parse_args()
    run_training(para_path=args.config, train_path=args.data_path, log_path=args.log_path)


if __name__ == '__main__':
    main()