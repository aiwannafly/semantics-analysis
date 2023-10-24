import torch
import torch.nn as nn
from transformers import AutoModel

import config


class TermRecognizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.MODEL_PATH)
        self.linear1 = nn.Linear(config.HIDDEN_SIZE, config.NUM_CLASSES + 1, device=config.DEVICE)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(50, config.NUM_CLASSES + 1, device=config.DEVICE)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.bert(input_ids, token_type_ids, attention_mask)
        x = x.last_hidden_state
        x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        x = self.softmax(x)

        return x
