import math
# from turtle import backward, forward

import torch
import torch.nn as nn
from torch.autograd import Function
# from torch.cuda.amp import custom_bwd, custom_fwd 
import atexit
import torch.nn.functional as F
from functools import partial


class py_FFMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, device, activation='relu', std=-1.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        # self.output_activation = convert_activation('none') # not supported currently
        self.output_activation = None

        self.tensorcore_width = 16

        # assert hidden_dim in [16, 32, 64, 128, 256], f"FFMLP only support hidden_dim in [16, 32, 64, 128, 256], but got {hidden_dim}"
        # assert input_dim > 0 and input_dim % 16 == 0, f"FFMLP input_dim should be 16 * m (m  > 0), but got {input_dim}"
        # assert output_dim <= 16, f"FFMLP current only supports output dim <= 16, but got {output_dim}"
        # assert num_layers >= 2, f"FFMLP num_layers should be larger than 2 (3 matmuls), but got {num_layers}"
        
        # pad output
        # self.padded_output_dim = int(math.ceil(output_dim / 16)) * 16

        # parameters (continuous in memory)
        self.Network = nn.Sequential().to(device)
        for i in range(self.num_layers):
            if i == 0:
                self.Network.add_module('Linear'+str(i), nn.Linear(self.input_dim, self.hidden_dim, bias=False).to(device))
                self.Network.add_module('Activation'+str(i), nn.ReLU().to(device))
            elif i == self.num_layers - 1:
                self.Network.add_module('Linear'+str(i), nn.Linear(self.hidden_dim, self.output_dim, bias=False).to(device))
                self.Network.add_module('Activation'+str(i), nn.ReLU().to(device))
            else:
                self.Network.add_module('Linear'+str(i), nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(device))
            self.Network.add_module('Activation', nn.ReLU().to(device))


        # self.num_parameters = hidden_dim * (input_dim + hidden_dim * (num_layers - 1) + self.padded_output_dim)
        # self.weights = nn.Parameter(torch.zeros(self.num_parameters))
        self.reset_parameters(std=std)
        # allocate streams
        
        # _backend.allocate_splitk(self.num_layers + 1)#bingo

        # register destructor
        #atexit.register(self.cleanup) # how to correctly clean? this gives CUDA Error: cudaEventDestroy(events[i]) failed with error context is destroyed


    # def cleanup(self):
    #     # destroy streams
    #     _backend.free_splitk()
    

    def __repr__(self):
        return f"FFMLP: input_dim={self.input_dim} output_dim={self.output_dim} hidden_dim={self.hidden_dim} num_layers={self.num_layers} activation={self.activation}"

    def weights_init(m, std):
        # if isinstance(m, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(m.weight)
        #     torch.nn.init.zero_(m.bias)
        torch.nn.init.uniform_(m, -std, std)


    def reset_parameters(self, std=-1.0):
        def weights_init(m, std):
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight, -std, std)
            
        torch.manual_seed(42)
        if std < 0.0:
            std = math.sqrt(3 / self.hidden_dim)
        func = partial(weights_init, std=std)
        # self.weights.data.uniform_(-std, std)
        # self.Network.weight.data.uniform_(-std, std)
        self.Network.apply(func)
    
    

    def forward(self, inputs):
        # inputs: [B, input_dim]
        # return: [B, outupt_dim]

        # if torch.isnan(inputs).any():
        #     raise ValueError("error!")
        B, C = inputs.shape
        #assert B >= 128 and B % 128 == 0, f"ffmlp batch size must be 128 * m (m > 0), but got {B}."

        # pad input
        # pad = 128 - (B % 128)
        # if pad > 0:
        #     inputs = torch.cat([inputs, torch.zeros(pad, C, dtype=inputs.dtype, device=inputs.device)], dim=0)

        # outputs = ffmlp_forward(inputs, self.weights, self.input_dim, self.padded_output_dim, self.hidden_dim, self.num_layers, self.activation, self.output_activation, not self.training, inputs.requires_grad)
        outputs = self.Network(inputs)

        # unpad output
        # if B != outputs.shape[0] or self.padded_output_dim != self.output_dim:
        #     outputs = outputs[:B, :self.output_dim]


        return outputs