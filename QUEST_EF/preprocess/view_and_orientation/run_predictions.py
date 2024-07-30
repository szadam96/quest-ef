import lightning as pl
import torch
from QUEST_EF.preprocess.view_and_orientation.utils.transforms import get_transforms
from QUEST_EF.preprocess.view_and_orientation.model.model import ModelV
from QUEST_EF.preprocess.view_and_orientation.model.architecture import ResNeXt50Module
from QUEST_EF.preprocess.view_and_orientation.dataset.image_dataset import EchoImageDataset
from QUEST_EF.preprocess.view_and_orientation.utils.callbacks import OrientationPredictionWriter, ViewPredicitonWriter

def run_classifier(config, data_csv, type_='view', devices='auto'):
    model = ModelV.load_from_checkpoint(config[type_]['checkpoint_path'], pytorch_module=ResNeXt50Module(config[type_]['out_dim']))
    transforms = get_transforms(config)
    dataset = EchoImageDataset(config['data_dir'], transform=transforms, data_csv=data_csv)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config[type_]['batch_size'], shuffle=False, num_workers=4)
    model.eval()
    if type_ == 'view':
        writer = ViewPredicitonWriter(data_csv, config[type_]['prediction_threshold'])
    elif type_ == 'orientation':
        writer = OrientationPredictionWriter(data_csv, config[type_]['prediction_threshold'])
    else:
        raise ValueError('type_ must be view or orientation')
    trainer = pl.Trainer(accelerator='cpu', devices=devices, callbacks=[writer])
    trainer.predict(model, dataloader)
