from torch.nn import (Sequential, Linear, ReLU, Sigmoid, Tanh,
                      Conv2d, MaxPool2d,
                      MSELoss, CrossEntropyLoss,
                      BatchNorm2d, BatchNorm1d, ModuleList)
from torchsummary import summary
import torch
import numpy as np

_activation = {'relu': ReLU,
               'tanh': Tanh,
               'sigmoid': Sigmoid}

def make_conv2d_model(input_shape, output_shape, params):
    batch_norm = params.get('use_batch_norm', False)

    layers = ModuleList()

    out_filters = params['initial_kernel_number']
    output = torch.zeros((2,) + input_shape)
    if params['n_conv_layers'] > 0:
        for ii in range(params['n_conv_layers']):
            if ii == 0:
                shp = output.shape[2:]
                kernel_size = params['input_kernel_size']
                kernel_size = (min(kernel_size[0], shp[0]),
                               min(kernel_size[1], shp[1]))
                layers.append(Conv2d(input_shape[0], out_filters,
                                     kernel_size))
                output = layers[-1].forward(output)
                layers.append(_activation[params['activation']]())
                output = layers[-1].forward(output)
                if batch_norm:
                    layers.append(BatchNorm2d(out_filters))
                    output = layers[-1].forward(output)
            else:
                in_filters = out_filters

                if params['conv_dim_change'] == 'double':
                    out_filters = out_filters * 2
                elif params['conv_dim_change'] == 'halve-first':
                    if ii == 0:
                        out_filters = out_filters // 2
                elif params['conv_dim_change'] == 'halve-last':
                    if ii == params['n_conv_layers']-2:
                        out_filters = out_filters // 2

                shp = output.shape[2:]
                kernel_size = params['conv_kernel_size']
                kernel_size = (min(kernel_size[0], shp[0]),
                                min(kernel_size[1], shp[1]))
                layers.append(Conv2d(in_filters, out_filters,
                                      kernel_size))
                output = layers[-1].forward(output)
                layers.append(_activation[params['activation']]())
                output = layers[-1].forward(output)
                if batch_norm:
                    layers.append(BatchNorm2d(out_filters))
                    output = layers[-1].forward(output)

                if params['pool_size'] is not None:
                    shp = output.shape[2:]
                    pool_size = params['pool_size']
                    pool_size = (min(kernel_size[0], shp[0]),
                                 min(kernel_size[1], shp[1]))
                    layers.append(MaxPool2d(pool_size))
                    output = layers[-1].forward(output)
    lin_layers = ModuleList()
    input_dim = np.prod(output.shape[1:])
    current_dim = params['dense_dim']
    if params['n_dense_layers'] == 0:
        return layers, None
    if params['n_dense_layers'] > 1:
        for ii in range(params['n_dense_layers']-1):
            if ii == 0:
                lin_layers.append(Linear(input_dim, current_dim))
            else:
                lin_layers.append(Linear(current_dim, current_dim))
            layers.append(_activation[params['activation']]())
            if batch_norm:
                lin_layers.append(BatchNorm1d(current_dim))
            if params['dense_dim_change'] != 'none':
                raise NotImplementedError
    else:
        current_dim = input_dim
    lin_layers.append(Linear(current_dim, output_shape))

    return layers, lin_layers

class mod(torch.nn.Module):
    def __init__(self, lay, lin):
        super().__init__()
        self.lay = lay
        self.lin = lin
    def forward(self, x):
        num = x.shape[0]
        for l in self.lay:
            x = l(x)
        if self.lin == None:
            return x
        x = x.view(num, -1)
        for l in self.lin:
            x = l(x)
        return x