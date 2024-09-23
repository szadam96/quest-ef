from pathlib import Path
from QUEST_EF.dataset.video_dataset import EchoVideoDataset
from QUEST_EF.models.model import EchoModel
from QUEST_EF.models.architectures import BiventricularModel
from QUEST_EF.utils.train_utils import get_augmentations, get_callbacks, get_model_config
from QUEST_EF.utils.utils import fix_seed
from QUEST_EF.utils.prediction_writer import PredictionWriter
import torch
import lightning as pl
from lightning.pytorch import loggers as pl_loggers
import tensorboardX
from torch.utils.data import DataLoader
from argparse import ArgumentParser


def run_prediction(model_checkpoint, root_dir, data_json, output_dir, accerator='cpu'):
    ckpt = torch.load(model_checkpoint, map_location='cpu')
    parameters = ckpt['hyper_parameters']
    
    # Set seed
    fix_seed(parameters['seed'])

    parameters['model']['ckpt_path'] = None

    augmentations = parameters['augmentations']
    dataloader_params = parameters['dataloader']

    transform = get_augmentations(augmentations)

    data_set = EchoVideoDataset(data_json,
                                root_dir=root_dir,
                                transform=transform['validation'],
                                include_binary_mask=dataloader_params['include_binary_mask'],
                                limit=dataloader_params.get('limit', None))

    dataloader = DataLoader(data_set, batch_size=dataloader_params['batch_size'], num_workers=8)

    tensorboard_logger = pl_loggers.TensorBoardLogger(output_dir)
    
    # Initialize model
    config = get_model_config(parameters['model'])
    m = BiventricularModel(config, model_checkpoint)
    model = EchoModel(parameters, model=m)

    callbacks = [PredictionWriter(output_dir=output_dir)]

    # Initialize trainer
    trainer = pl.Trainer(accelerator=accerator,
                         devices=1,
                         max_epochs=parameters['epochs'],
                         logger=tensorboard_logger,
                         callbacks=callbacks,
                         precision=32)

    # Train
    trainer.predict(model, dataloader)


def main():
    args = ArgumentParser()
    args.add_argument('--model_path', type=str, required=True)
    args.add_argument('--root_dir', type=str, required=True)
    args.add_argument('--data_json', type=str, required=True)
    args.add_argument('--output_dir', type=str, required=True)
    args.add_argument('--accelerator', type=str, default='cpu')

    args = args.parse_args()

    run_prediction(args.model_path, Path(args.root_dir), args.data_json, args.output_dir, args.accelerator)


if __name__ == '__main__':
    main()
