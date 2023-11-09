import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils import *
from collections import defaultdict
from lib.models.layers import LocallyConnected2d
from lib.models.layers.feature_correction import *

class BidirectionalModel(nn.Module):
    """Take sequence of poses to predict: \n
    `walk_speed` : [ x, y, z ](normalized with leg length)\n
    `step_length`: [ l ] (does not differentiate the left/right)
    """
    def __init__(self, 
        seqlen, 
        # h_size=100,
        input_size=128,# per joint
        num_joints=24, 
        num_outputs=3,
        estime_phase=True,
        fc_size=40,
        num_layers=2,
        use_pareFeat=False,
        ):
        super().__init__()

        self.estim_phase = estime_phase
        multiple = 3
        self.num_outputs = num_outputs
        h_size = 300 if use_pareFeat else 100
        fc_size = 100 if use_pareFeat else 40
        self.num_layers = num_layers
        self.use_pareFeat = use_pareFeat
        # if not self.use_pareFeat:
        #     self.h0 = nn.Parameter(torch.zeros(num_layers*2, 1, h_size).normal_(std=0.01), requires_grad=True)
        if not self.use_pareFeat:
            self.input_size = num_joints*multiple + 3
        else:
            self.input_size = input_size*num_joints
            self.dropout = nn.Dropout(0.2)
            cparam_out_channel = 128
            self.cparam_mpl = LocallyConnected2d(
                in_channels=3,
                out_channels=cparam_out_channel,
                output_size=[num_joints,1],
                kernel_size=1,
                stride=1,
            )

        self.rnn = nn.GRU(
            input_size=self.input_size, 
            hidden_size=h_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True,
        )
        if num_outputs>0:
            # split speed and step estimations 
            self.speed_mlp = nn.Sequential(
                nn.Linear(h_size*2*num_layers, fc_size),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Linear(fc_size, 1)
            )
            self.step_mlp = nn.Sequential(
                nn.Linear(h_size*2*num_layers, fc_size),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Linear(fc_size, 2)
            ) 

        if estime_phase:
            self.phase_mlp = nn.Sequential(
                nn.Linear(h_size*2, fc_size), # x2 for bidirectional
                nn.LeakyReLU(0.05, inplace=True),
                nn.Linear(fc_size, 4),
                nn.Tanh()
            )
    
    def forward(self, x, cparams=None):
        # ===== preprocess cparams with locallyConnectedLayer ===== #
        if self.use_pareFeat:
            assert cparams is not None
            b, n, cf = cparams.shape
            xc = self.cparam_mpl(cparams.reshape(b*n,cf,1,1).expand(b*n,cf,24,1)).reshape(b,n,-1)
            # residual add
            x = x + self.dropout(xc)

        x, h = self.rnn(x,)
        h = h.permute(1,0,2)
        h = h.reshape(h.shape[0],-1)
        
        if self.num_outputs:
            y1 = self.speed_mlp(h)
            y2 = self.step_mlp(h)
            y = torch.cat((y1, y2), dim=-1)
        else:
            y = None
        
        if self.estim_phase:
            p = self.phase_mlp(x)
            return y, p, xc
        else:
            return y, None, xc