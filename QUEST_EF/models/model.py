import os
import torch
import torch.nn.functional as F
import lightning as pl
from torchmetrics import MeanSquaredError
from torchmetrics import MeanAbsoluteError, R2Score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from QUEST_EF.models.architectures import CustomVideoMAEForRegression, BiventricularModel
from QUEST_EF.utils.train_utils import get_optimizer, get_scheduler, get_regressor_model, get_model_config, get_weighted_mae, get_weighted_mse
from sklearn.metrics import r2_score

class EchoModel(pl.LightningModule):
    '''Pytorch Lightning model for the automated LVEF prediction task
    
    Attributes
    ----------
    config : dict
        Dictionary containing the configuration for the model
    '''
    def __init__(self, config, model=None, **kwargs):
        super().__init__()
        self.config = config
        self.lr = self.config['training']['learning_rate']
        self.mae_config = get_model_config(config['model'])
        if not model:
            self.model = get_regressor_model(config)
            if config['training'].get('ckpt_path', None) is not None:
                self.load_pretrain_state_dict(config['model']['ckpt_path'])
            if config['training'].get('freeze_encoder', False):
                self.freeze_encoder()
        else:
            self.model = model

        self.to_predict = config['training'].get('to_predict', ['LVEF', 'RVEF'])

        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        self.use_loss = config['training'].get('loss', 'mse')
        self.weighted_loss = config['training'].get('weighted', False)
        weight_func = kwargs.get('weight_func', lambda x: 1)
        if self.use_loss == 'mse':
            if self.weighted_loss:        
                self.loss_func = get_weighted_mse(weight_function=weight_func)
            else:
                self.loss_func = self.mse
        elif self.use_loss == 'mae':
            if self.weighted_loss:
                self.loss_func = get_weighted_mae(weight_function=weight_func)
            else:
                self.loss_func = self.mae
        elif self.use_loss == 'r2':
            self.loss_func = self.r2
            if self.weighted_loss:
                raise ValueError('R2 score cannot be weighted')
        self.best_loss = 1000
        self.save_hyperparameters(self.config)
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def freeze_encoder(self):
        '''Freeze the encoder'''
        for param in self.model.videomae.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        '''Unfreeze the encoder'''
        for param in self.model.videomae.parameters():
            param.requires_grad = True

    def load_pretrain_state_dict(self, ckpt_path, strict=False):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt['state_dict']
        for key in list(state_dict.keys()):
            if 'model.model.' in key:
                new_key = key.replace('model.model.', '')
                val = state_dict.pop(key)
                if 'videomae' in key:
                    state_dict[new_key] = val
        self.model.load_state_dict(state_dict, strict=strict)
        print('Loaded model from checkpoint')

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        '''Configure the optimizer and scheduler'''
        maximize = False
        if self.use_loss == 'r2':
            maximize = True
        separate_params = self.config['training'].get('separate_lr_encoder', False)
        optimizer = get_optimizer(self.config['training'], self, separate_params, maximize=maximize)
        scheduler = get_scheduler(self.config['training'], optimizer)
        return [optimizer], [scheduler]
    
    def common_step(self, batch, batch_idx):
        '''Common step for the model'''
        x = batch['input_tensor']
        binary_mask = batch['binary_mask']
        if binary_mask is not None:
            binary_mask_pos = self.binary_mask_pos(binary_mask)
        rvef_true = batch['RVEF']
        lvef_true = batch['LVEF']
        patient_id = batch['patient_id']
        dicom_id = batch['dicom_id']
        if isinstance(self.model, CustomVideoMAEForRegression) or isinstance(self.model, BiventricularModel):
            pred = self.model(x, binary_mask_pos)
        else:
            pred = self(x)
        rvef_pred, lvef_pred = pred['rv_ef'].squeeze(), pred['lv_ef'].squeeze()
        loss = 0
        if 'RVEF' in self.to_predict:
            loss += self.loss_func(rvef_pred, rvef_true)
        if 'LVEF' in self.to_predict:
            loss += self.loss_func(lvef_pred, lvef_true)
        if not 'RVEF' in self.to_predict and not 'LVEF' in self.to_predict:
            raise ValueError('No task to predict')
        if len(rvef_pred.size()) == 0:
            rvef_pred = rvef_pred.unsqueeze(0)
        if len(lvef_pred.size()) == 0:
            lvef_pred = lvef_pred.unsqueeze(0)
        output = {'loss': loss, 'patient_id': patient_id, 'dicom_id': dicom_id, 'rvef_pred': rvef_pred, 'rvef_true': rvef_true, 'lvef_pred': lvef_pred, 'lvef_true': lvef_true}
        return output
    

    def training_step(self, batch, batch_idx):
        '''Training step for the model'''
        output = self.common_step(batch, batch_idx)
        loss = output['loss']
        
        self.train_outputs.append({
            'loss': loss.detach()
        })
        return loss

    def on_train_epoch_end(self):
        '''Training epoch end for the model'''
        outputs = self.train_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss.detach(), prog_bar=False, logger=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[-1]['lr'], logger=True)
        self.train_outputs = []
        #if self.current_epoch == 10 and not self.config['training'].get('freeze_encoder', False):
        #    self.unfreeze_encoder()
    
    def validation_step(self, batch, batch_idx):
        '''Validation step for the model'''
        output = self.common_step(batch, batch_idx)
        self.val_outputs.append({
            'loss': output['loss'].detach(),
            'patient_id': output['patient_id'],
            'dicom_id': output['dicom_id'],
            'rvef_pred': output['rvef_pred'].detach().cpu(),
            'rvef_true': output['rvef_true'].detach().cpu(),
            'lvef_pred': output['lvef_pred'].detach().cpu(),
            'lvef_true': output['lvef_true'].detach().cpu()
        })
        
    

    #save the model after each epoch if the validation loss is lower than the previous one and log the average validation loss and mae of every head
    def on_validation_epoch_end(self):
        '''Validation epoch end for the model'''
        outputs = self.val_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).detach().mean()

        self.log_metrics(outputs)
        self.val_outputs = []
    
    def test_step(self, batch, batch_idx):
        '''Test step for the model'''
        output = self.common_step(batch, batch_idx)
        self.test_outputs.append({
            'loss': output['loss'].detach(),
            'patient_id': output['patient_id'],
            'dicom_id': output['dicom_id'],
            'rvef_pred': output['rvef_pred'].detach().cpu(),
            'rvef_true': output['rvef_true'].detach().cpu(),
            'lvef_pred': output['lvef_pred'].detach().cpu(),
            'lvef_true': output['lvef_true'].detach().cpu()
        })
        

    def on_test_epoch_end(self):
        '''Test epoch end for the model'''
        outputs = self.test_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).detach().mean()
        self.log_metrics(outputs)
        self.test_outputs = []


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        '''Predict step for the model'''
        output = self.common_step(batch, batch_idx)
        return output
        


    def log_metrics(self, outputs):
        '''Log the metrics for the model'''
        rvef_preds = torch.cat([x['rvef_pred'] for x in outputs])
        rvef_trues = torch.cat([x['rvef_true'] for x in outputs])
        lvef_preds = torch.cat([x['lvef_pred'] for x in outputs])
        lvef_trues = torch.cat([x['lvef_true'] for x in outputs])
        mse_lvef = self.mse(lvef_preds, lvef_trues)
        mae_lvef = self.mae(lvef_preds, lvef_trues)
        r2_lvef = self.r2(lvef_preds, lvef_trues)

        mse_rvef = self.mse(rvef_preds, rvef_trues)
        mae_rvef = self.mae(rvef_preds, rvef_trues)
        r2_rvef = self.r2(rvef_preds, rvef_trues)

        self.log('mse_lvef', mse_lvef, prog_bar=False, logger=True, on_epoch=True)
        self.log('mae_lvef', mae_lvef, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_lvef', r2_lvef, prog_bar=False, logger=True, on_epoch=True)

        self.log('mse_rvef', mse_rvef, prog_bar=False, logger=True, on_epoch=True)
        self.log('mae_rvef', mae_rvef, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_rvef', r2_rvef, prog_bar=False, logger=True, on_epoch=True)
        

        validation_df = pd.DataFrame({'patient_id': np.concatenate([x['patient_id'] for x in outputs]),
                                        'dicom_id': np.concatenate([x['dicom_id'] for x in outputs]),
                                        'lvef_true': [x for x in lvef_trues.numpy()],
                                        'lvef_pred': [x for x in lvef_preds.numpy()],
                                        'rvef_true': [x for x in rvef_trues.numpy()],
                                        'rvef_pred': [x for x in rvef_preds.numpy()]})
        
        lvef_true_video = torch.tensor(validation_df['lvef_true'].to_numpy())
        lvef_pred_video = torch.tensor(validation_df['lvef_pred'].to_numpy())
        rvef_true_video = torch.tensor(validation_df['rvef_true'].to_numpy())
        rvef_pred_video = torch.tensor(validation_df['rvef_pred'].to_numpy())
        
        os.makedirs(self.logger.log_dir + '/predictions', exist_ok=True)
        validation_df.to_csv(self.logger.log_dir + f'/predictions/pred_{self.trainer.current_epoch}.csv', index=False)

        val_df_patient = validation_df.groupby('patient_id').mean()
        lvef_true_patient = torch.tensor(val_df_patient['lvef_true'].to_numpy())
        lvef_pred_patient = torch.tensor(val_df_patient['lvef_pred'].to_numpy())
        mse_lvef_patient = self.mse(lvef_pred_patient, lvef_true_patient)
        mae_lvef_patient = self.mae(lvef_pred_patient, lvef_true_patient)
        r2_lvef_patient = self.r2(lvef_pred_patient, lvef_true_patient)
        try:
            sample_patient_counts = val_df_patient['lvef_true'].value_counts(bins=10)
            sample_patient_weights = val_df_patient['lvef_true'].map(1/sample_patient_counts)
            r2_lvef_patient_weighted = r2_score(lvef_true_patient, lvef_pred_patient, sample_weight=sample_patient_weights)
        except:
            r2_lvef_patient_weighted = 0

        rvef_true_patient = torch.tensor(val_df_patient['rvef_true'].to_numpy())
        rvef_pred_patient = torch.tensor(val_df_patient['rvef_pred'].to_numpy())
        mse_rvef_patient = self.mse(rvef_pred_patient, rvef_true_patient)
        mae_rvef_patient = self.mae(rvef_pred_patient, rvef_true_patient)
        r2_rvef_patient = self.r2(rvef_pred_patient, rvef_true_patient)
        try:
            sample_patient_counts = val_df_patient['rvef_true'].value_counts(bins=10)
            sample_patient_weights = val_df_patient['rvef_true'].map(1/sample_patient_counts)
            r2_rvef_patient_weighted = r2_score(rvef_true_patient, rvef_pred_patient, sample_weight=sample_patient_weights)
        except:
            r2_rvef_patient_weighted = 0
        self.log('mse_lvef_patient', mse_lvef_patient, prog_bar=False, logger=True, on_epoch=True)
        self.log('mae_lvef_patient', mae_lvef_patient, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_lvef_patient', r2_lvef_patient, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_lvef_patient_weighted', r2_lvef_patient_weighted, prog_bar=False, logger=True, on_epoch=True)

        self.log('mse_rvef_patient', mse_rvef_patient, prog_bar=False, logger=True, on_epoch=True)
        self.log('mae_rvef_patient', mae_rvef_patient, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_rvef_patient', r2_rvef_patient, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_rvef_patient_weighted', r2_rvef_patient_weighted, prog_bar=False, logger=True, on_epoch=True)

        val_df_dicom = validation_df.groupby(['patient_id', 'dicom_id']).mean()
        lvef_true_dicom = torch.tensor(val_df_dicom['lvef_true'].to_numpy())
        lvef_pred_dicom = torch.tensor(val_df_dicom['lvef_pred'].to_numpy())
        mse_lvef_dicom = self.mse(lvef_pred_dicom, lvef_true_dicom)
        mae_lvef_dicom = self.mae(lvef_pred_dicom, lvef_true_dicom)
        r2_lvef_dicom = self.r2(lvef_pred_dicom, lvef_true_dicom)
        try:
            sample_dicom_counts = val_df_dicom['lvef_true'].value_counts(bins=10)
            sample_dicom_weights = val_df_dicom['lvef_true'].map(1/sample_dicom_counts)
            r2_lvef_dicom_weighted = r2_score(lvef_true_dicom, lvef_pred_dicom, sample_weight=sample_dicom_weights)
        except:
            r2_lvef_dicom_weighted = 0
        rvef_true_dicom = torch.tensor(val_df_dicom['rvef_true'].to_numpy())
        rvef_pred_dicom = torch.tensor(val_df_dicom['rvef_pred'].to_numpy())
        mse_rvef_dicom = self.mse(rvef_pred_dicom, rvef_true_dicom)
        mae_rvef_dicom = self.mae(rvef_pred_dicom, rvef_true_dicom)
        r2_rvef_dicom = self.r2(rvef_pred_dicom, rvef_true_dicom)
        try:
            sample_dicom_counts = val_df_dicom['rvef_true'].value_counts(bins=10)
            sample_dicom_weights = val_df_dicom['rvef_true'].map(1/sample_dicom_counts)
            r2_rvef_dicom_weighted = r2_score(rvef_true_dicom, rvef_pred_dicom, sample_weight=sample_dicom_weights)
        except:
            r2_rvef_dicom_weighted = 0
        self.log('mse_lvef_dicom', mse_lvef_dicom, prog_bar=False, logger=True, on_epoch=True)
        self.log('mae_lvef_dicom', mae_lvef_dicom, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_lvef_dicom', r2_lvef_dicom, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_lvef_dicom_weighted', r2_lvef_dicom_weighted, prog_bar=False, logger=True, on_epoch=True)

        self.log('mse_rvef_dicom', mse_rvef_dicom, prog_bar=False, logger=True, on_epoch=True)
        self.log('mae_rvef_dicom', mae_rvef_dicom, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_rvef_dicom', r2_rvef_dicom, prog_bar=False, logger=True, on_epoch=True)
        self.log('r2_rvef_dicom_weighted', r2_rvef_dicom_weighted, prog_bar=False, logger=True, on_epoch=True)

        self.log('val_r2_patient', (r2_lvef_patient + r2_rvef_patient)/2, prog_bar=False, logger=True, on_epoch=True)
        self.log('val_r2_dicom', (r2_lvef_dicom + r2_rvef_dicom)/2, prog_bar=False, logger=True, on_epoch=True)

        self.logger.experiment.add_figure('predictions vs targets for lvef patient', 
                                          self.plot_predictions(lvef_pred_patient, lvef_true_patient),
                                          global_step=self.current_epoch)

        self.logger.experiment.add_figure('predictions vs targets for rvef patient',
                                          self.plot_predictions(rvef_pred_patient, rvef_true_patient),
                                          global_step=self.current_epoch)

        self.logger.experiment.add_figure('predictions vs targets for lvef dicom',
                                          self.plot_predictions(lvef_pred_dicom, lvef_true_dicom),
                                          global_step=self.current_epoch)

        self.logger.experiment.add_figure('predictions vs targets for rvef dicom',
                                          self.plot_predictions(rvef_pred_dicom, rvef_true_dicom),
                                          global_step=self.current_epoch)

        self.logger.experiment.add_figure('predictions vs targets for lvef video',
                                          self.plot_predictions(lvef_pred_video, lvef_true_video),
                                          global_step=self.current_epoch)

        self.logger.experiment.add_figure('predictions vs targets for rvef video',
                                          self.plot_predictions(rvef_pred_video, rvef_true_video),
                                          global_step=self.current_epoch)

        plt.close('all')

    def plot_predictions(self, y_preds, y_trues):
        '''Plot the predictions vs targets'''
        fig, ax = plt.subplots(1, 1)
        ax.scatter(y_preds, y_trues)
        ax.plot([int(min(min(y_preds), min(y_trues))), int(max(max(y_preds), max(y_trues)))],
                 [int(min(min(y_preds), min(y_trues))), int(max(max(y_preds), max(y_trues)))], c='r')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Predictions vs Targets')
        return fig
    
    def binary_mask_pos(self, binary_mask):
        batch_size = binary_mask.shape[0]
        mask = torch.any(binary_mask, dim=0)
        mask = ~F.max_pool2d(mask.unsqueeze(0).unsqueeze(0).float(), self.mae_config.patch_size).squeeze().bool()
        mask = torch.stack([mask for _  in range(self.mae_config.num_frames // self.mae_config.tubelet_size)], dim=0).flatten()
        return torch.stack([mask for _ in range(batch_size)], dim=0)