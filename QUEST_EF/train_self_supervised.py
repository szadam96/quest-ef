import os
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset.self_supervised_dm import EchoVideoDataModule
from models.model_pretraining import EchoModel
from utils.train_utils import get_callbacks
from utils.utils import fix_seed

import torch.nn as nn
import torch.nn.functional as F


def run_training(load_model=False, model_path=None, para_path=None):
    '''Run training for the model
    
    Parameters
    ----------
    load_model : bool
        Whether to load a model from a checkpoint
    model_path : Path
        Path to the checkpoint
    para_path : str
        Path to the yaml file containing the parameters
        
    Returns
    -------
    None
    '''

    # Load augmentation yaml
    with open(para_path, 'r') as f:
        parameters = yaml.safe_load(f)

    # Set seed
    fix_seed(parameters['seed'])

    augmentations = parameters['augmentations']
    dataloader_params = parameters['dataloader']

    train_path = 'self-suervised_data/train_set_multilabel.json'
    val_path = 'self-suervised_data/test_set_multilabel.json'

    dm = EchoVideoDataModule(train_path, val_path,
                             augmentations=augmentations,
                             mae_config=parameters['model'],
                             **dataloader_params)

    tensorboard_logger = TensorBoardLogger('logs_pretraining/')

    # Initialize model
    model = EchoModel(parameters)

    # Initialize callback
    callbacks = get_callbacks(parameters['callbacks'])

    # Initialize trainer
    trainer = pl.Trainer(accelerator='gpu',
                         devices=2,
                         max_epochs=parameters['epochs'],
                         logger=tensorboard_logger,
                         callbacks=callbacks,
                         precision='16-mixed')

    # Train
    if load_model:
        trainer.fit(model, dm, ckpt_path=model_path)
    else:
        trainer.fit(model, dm)


if __name__ == '__main__':
    run_training(load_model=True, para_path='parameters_self_supervised.yaml', model_path='logs_pretraining/best_loss.ckpt')
