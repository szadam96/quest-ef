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
        temp_df = pd.DataFrame(list(zip(dicom_ids, pred_probas[:, 0], pred_probas[:, 1], pred_probas[:, 2])), columns=['dicom_id', 'other_pred', 'a4ch_rf_pred', 'a4ch_pred'])
        temp_df = temp_df.groupby('dicom_id').mean().reset_index()
        temp_df['view_pred'] = temp_df.apply(lambda x: 'other' if 
                                        x['other_pred'] > self.prediction_threshold else
                                          ('a4ch(rv focused)' if x['a4ch_rf_pred'] > x['a4ch_pred'] else 'a4ch'), axis=1)
        self.df = pd.merge(self.df, temp_df[['dicom_id', 'view_pred', 'other_pred', 'a4ch_rf_pred', 'a4ch_pred']], on='dicom_id', how='outer')
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
        
    def write_on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]):
        pred_probas, dicom_ids = zip(*predictions)
        pred_probas = np.concatenate(pred_probas)
        dicom_ids = np.concatenate(dicom_ids)
        temp_df = pd.DataFrame(list(zip(dicom_ids, pred_probas[:, 0])), columns=['dicom_id', 'orientation_pred_proba'])
        
        temp_df = temp_df.groupby('dicom_id').mean().reset_index()
        temp_df['orientation_pred'] = temp_df['orientation_pred_proba'].apply(lambda x: 'mayo' if x > self.prediction_threshold else 'stanford')
        self.df = pd.merge(self.df, temp_df[['dicom_id', 'orientation_pred']], on='dicom_id', how='outer')
        self.df.to_csv(self.output_csv, index=False)
        
            
            