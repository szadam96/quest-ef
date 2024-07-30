from transformers import VideoMAEForPreTraining
from transformers.models.videomae.modeling_videomae import VideoMAEForPreTrainingOutput, VideoMAEPreTrainedModel
from transformers.models.videomae.modeling_videomae import VideoMAEModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional   

class CustomVideoMAEForRegression(VideoMAEPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.videomae = VideoMAEModel(config)

        # Classifier head
        if 'dropout' in kwargs:
            dropout = kwargs['dropout']
        else:
            dropout = 0.2
        self.lvef_head = RegressorHead(config.hidden_size, 1, dropout=dropout)
        self.rvef_head = RegressorHead(config.hidden_size, 1, dropout=dropout)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        outputs = self.videomae(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )

        sequence_output = outputs[0]

        lvef_pred = self.lvef_head(sequence_output)
        rvef_pred = self.rvef_head(sequence_output)

        loss = None

        return {
            "lv_ef": lvef_pred,
            "rv_ef": rvef_pred,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }


class RegressorHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super().__init__()
        self.fc_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.fc_norm(x.mean(1))
        x = self.dropout(F.relu(self.fc1(self.dropout(x))))
        return self.fc2(x)
    
class BiventricularModel(nn.Module):
    def __init__(self, config, lvef_ckpt_path, rvef_ckpt_path):
        super().__init__()
        self.lvef_model = CustomVideoMAEForRegression(config)
        self.rvef_model = CustomVideoMAEForRegression(config)
        self._load_from_state_dict(self.lvef_model, lvef_ckpt_path)
        self._load_from_state_dict(self.rvef_model, rvef_ckpt_path)

    def _load_from_state_dict(self, model, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict']
        for key in list(state_dict.keys()):
            if 'model.' in key:
                new_key = key.replace('model.', '')
                val = state_dict.pop(key)
                state_dict[new_key] = val
        model.load_state_dict(state_dict)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        lvef_output = self.lvef_model(pixel_values, bool_masked_pos, output_attentions, output_hidden_states)
        rvef_output = self.rvef_model(pixel_values, bool_masked_pos, output_attentions, output_hidden_states)
        return {
            "lv_ef": lvef_output["lv_ef"],
            "rv_ef": rvef_output["rv_ef"],
        }
    