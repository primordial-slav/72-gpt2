from dataclasses import dataclass
from torch.nn import nn
import torch
from torch.nn import functional as F


@dataclass
class GPTConfig:
    blocksize : int = 256
    vocabsize : int = 65
    n_layer : int = 6
    n_head : int = 6
    n_embd: int = 384


class GPT(nn.Module):
    def __init__(self,config):
        super.__init__()
        self.config = config