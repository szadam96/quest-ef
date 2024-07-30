import os
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd

import lightning as pl
from lightning.pytorch.callbacks import BasePredictionWriter

class ViewPredicitonWriter(BasePredictionWriter):
    def __init__(self, output_csv: str = None, prediction_threshold: float = 0.5):
        super().__init__(write_interval='epoch')
        self.prediction_threshold = prediction_threshold
        self.output_csv = output_csv
        if os.path.exists(self.output_csv):
            self.df = pd.read_csv(self.output_csv)
            if 'dicom_id' not in self.df.columns:
                raise ValueError('output_csv must have a column named dicom_id')

        else:
            self.df = pd.DataFrame(columns=['dicom_id'])

    def write_on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]):
        pred_probas, dicom_ids = zip(*predictions)
        pred_probas = np.concatenate(pred_probas)
        dicom_ids = np.concatenate(dicom_ids)
        temp_df = pd.DataFrame(columns=['dicom_id', 'other_pred', 'a4ch_rf_pred', 'a4ch_pred'])
        for pred_proba, dicom_id in zip(pred_probas, dicom_ids):
            pred_other = pred_proba[0]
            pred_rf = pred_proba[1]
            pred_standard = pred_proba[2]
            if pred_rf + pred_standard < self.prediction_threshold:
                pred_view = 'other'
            elif pred_rf > pred_standard:
                pred_view = 'a4ch(rv focused)'
            else:
                pred_view = 'a4ch'
            temp_df = pd.concat([temp_df, pd.DataFrame([[dicom_id, pred_other, pred_rf, pred_standard]], columns=['dicom_id', 'other_pred', 'a4ch_rf_pred', 'a4ch_pred'])])
        
        temp_df = temp_df.groupby('dicom_id').mean().reset_index()
        temp_df['view'] = temp_df.apply(lambda x: 'other' if 
                                        x['other_pred'] > self.prediction_threshold else
                                          ('a4ch(rv focused)' if x['a4ch_rf_pred'] > x['a4ch_pred'] else 'a4ch'), axis=1)
        self.df = pd.merge(self.df, temp_df[['dicom_id', 'view']], on='dicom_id', how='outer')
        self.df.to_csv(self.output_csv, index=False)


class OrientationPredictionWriter(BasePredictionWriter):
    def __init__(self, output_csv: str = None, prediction_threshold: float = 0.5):
        super().__init__(write_interval='epoch')
        self.prediction_threshold = prediction_threshold
        self.output_csv = output_csv
        if not os.path.exists(self.output_csv):
            raise ValueError(f'{self.output_csv} does not exist.')
        self.df = pd.read_csv(self.output_csv)
        if 'dicom_id' not in self.df.columns:
            raise ValueError('output_csv must have a column named dicom_id')
        if 'view' not in self.df.columns:
            raise ValueError('output_csv must have a column named view. Try running the view classifier first.')

    def write_on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]):
        pred_probas, dicom_ids = zip(*predictions)
        pred_probas = np.concatenate(pred_probas)
        dicom_ids = np.concatenate(dicom_ids)
        temp_df = pd.DataFrame(columns=['dicom_id','orientation_pred'])
        for pred_proba, dicom_id in zip(pred_probas, dicom_ids):
            pred_mayo = pred_proba[0]
            
            temp_df = pd.concat([temp_df, pd.DataFrame([[dicom_id, pred_mayo]], columns=['dicom_id', 'orientation_pred'])])
        
        temp_df = temp_df.groupby('dicom_id').mean().reset_index()
        temp_df['orientation'] = temp_df['orientation_pred'].apply(lambda x: 'mayo' if x > self.prediction_threshold else 'stanford')
        self.df = pd.merge(self.df, temp_df[['dicom_id', 'orientation']], on='dicom_id', how='outer')
        self.df.to_csv(self.output_csv, index=False)
        
            
            