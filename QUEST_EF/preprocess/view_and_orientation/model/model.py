from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import lightning as pl
import pandas as pd

class ModelV(pl.LightningModule):
    def __init__(self, pytorch_module):
        super().__init__()
        self.pytorch_module = pytorch_module
        self.out_dim = pytorch_module.out_dim

    def forward(self, x):
        return self.pytorch_module(x)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x = batch['input_tensor']
        dicom_id = batch['dicom_id']
        y_hat = self.forward(x)
        pred_proba = torch.nn.functional.softmax(y_hat, dim=1)
        pred_proba = pred_proba.detach().cpu().numpy()
        return pred_proba, dicom_id