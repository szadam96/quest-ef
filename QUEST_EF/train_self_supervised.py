import os
from argparse import ArgumentParser
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


def run_training(para_path=None, data_path=None, log_path='logs/'):
    '''Run training for the model
    
    Parameters
    ----------
    para_path : str
        Path to the yaml file containing the parameters
    data_path : str
        Path to the data
    log_path : str
        Path to the logs
        
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

    data_path = Path(data_path)

    train_path = list(data_path.glob('train_set*.json'))[0]
    val_path = list(data_path.glob('val_set*.json'))[0]

    dm = EchoVideoDataModule(train_path, val_path,
                             augmentations=augmentations,
                             mae_config=parameters['model'],
                             **dataloader_params)

    tensorboard_logger = TensorBoardLogger(log_path, name='self_supervised')

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
    trainer.fit(model, dm)

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--log_path', type=str, default='logs/')
    args = parser.parse_args()

    run_training(para_path=args.config, data_path=args.data_path, log_path=args.log_path)

if __name__ == '__main__':
    main()
