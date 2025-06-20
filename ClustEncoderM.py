import torch
import torch.nn as nn
from typing import Iterable, List, Optional

class ClustAutoEncoder(nn.Module):
    def __init__(
            self, 
            dims: Iterable[int],
            activation=nn.ReLU(True),
            final_encoder_activation=nn.ReLU(True),
            final_decoder_activation = nn.ReLU(True),
            dropout=0.1,
            ):
        super().__init__()
        self.encoder = ClusterEncoder(dims, activation, final_encoder_activation, dropout)
        self.decoder = ClusterDecoder(dims[::-1], activation, final_decoder_activation, dropout)        

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output
    
class ClusterEncoder(nn.Module):
    def __init__(self,
            dims: Iterable[int],
            activation=nn.ReLU(True),
            final_encoder_activation=nn.ReLU(True),
            dropout=0.1
            ):
        super().__init__()
        encoder_layer = self._add_linear_layer(dims[:-1], activation, dropout)
        last_encoder_layer = self._add_linear_layer([dims[-2], dims[-1]], final_encoder_activation, dropout=None)
        encoder_layer.extend(last_encoder_layer)

        self.encoder = nn.Sequential(*encoder_layer)
        self.encoder.apply(self._init_weight)

    def _add_linear_layer(self, dims, activation, dropout):
        layers=[]
        for idx in range(len(dims)-1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.LayerNorm(dims[idx + 1]))
            if activation is not None:
                layers.append(activation)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        return layers
    
    def _init_weight(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # follow paper setting
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        return z

class ClusterDecoder(nn.Module):
    def __init__(self,
            dims: Iterable[int],
            activation=nn.ReLU(True),
            final_decoder_activation=nn.ReLU(True),
            dropout=0.1
            ):
        super().__init__()
        decoder_layer = self._add_linear_layer(dims[:-1], activation, dropout)
        last_encoder_layer = self._add_linear_layer([dims[-2], dims[-1]], final_decoder_activation, dropout=None)
        decoder_layer.extend(last_encoder_layer)

        self.decoder = nn.Sequential(*decoder_layer)
        self.decoder.apply(self._init_weight)

    def _add_linear_layer(self, dims, activation, dropout):
        layers=[]
        for idx in range(len(dims)-1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.LayerNorm(dims[idx + 1]))
            if activation is not None:
                layers.append(activation)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        return layers
    
    def _init_weight(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # follow paper setting
            nn.init.constant_(layer.bias, 0)

    def forward(self, z):
        output = self.decoder(z)
        return output

    