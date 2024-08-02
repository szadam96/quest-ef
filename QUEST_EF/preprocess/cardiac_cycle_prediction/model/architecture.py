import sys

from QUEST_EF.preprocess.cardiac_cycle_prediction.utils.utils import get_class
# bruh
from QUEST_EF.preprocess.cardiac_cycle_prediction.model.UVT import get_model

from torch import nn

def UVT(*args, **kwargs):
    return get_model(*args, **kwargs)

def create_model(model_config):
    model_class = get_class(model_config['name'], modules=['QUEST_EF.preprocess.cardiac_cycle_prediction.model.architecture'])
    model_config.pop('name')
    return model_class(**model_config)
