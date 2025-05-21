import torch
import torch.nn as nn
from typing import Iterable, List, Optional

class ClustEncoder(nn.Module):
    def __init__(
            self, 
            dims: Iterable[int],
            activation=nn.ReLU(True),
            final_activation=nn.ReLU(True),
            dropout=0.1,
            num_cluster=10,
            ):
        super.__init__()
        encoder_layer = self._add_linear_layer(dims[:-1], activation, dropout)
        last_layer = self._add_linear_layer([dims[-2], dims[-1]], final_activation, dropout=None)
        encoder_layer.extend(last_layer)

        self.encoder = nn.Sequential(*encoder_layer)
        self.encoder.apply(self._init_weight)

        # self.assignment = SoftClusterAssignment(num_cluster, dims[-1])
        self.save_hyperparameters()



    def _add_linear_layer(dims, activation, dropout):
        layers=[]
        for idx in range(len(dims)-1):
            layers.append(dims[idx], dims[idx + 1])
            layers.append(activation)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        return layers
    
    def _init_weight(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # follow paper setting
            nn.init.constant_(layer.bias, 0)


    def forward(self, x, centroid):
        z = self.encoder(x)
        return z
    
    
    



# class SoftClusterAssignment(nn.Module):
#     def __init__(
#         self,
#         num_cluster: int,
#         hidden_dim: int,
#         alpha: float = 1.0,
#         centroid: torch.tensor = None,
#     ):
#         super.__init__()
#         self.num_cluster = num_cluster
#         self.hidden_dim = hidden_dim
#         self.alpha = alpha

#         if centroid is None:
#             initial_centroid = torch.zeros(
#                 self.num_cluster, self.hidden_dim, dtype=torch.float
#             )
#             nn.init.normal_(initial_centroid, mean=0.0, std=0.01)
#         else:
#             initial_centroid = centroid
#         self.centroid = initial_centroid.cuda()  

#     def forward(self, z):
#         z = z.cuda()
#         diff = torch.sum((z.unsqueeze(1) - self.centroid) ** 2, 2)
#         numerator = 1.0 / (1.0 + (diff / self.alpha))
#         power = (self.alpha + 1.0) / 2
#         numerator = numerator ** power
#         q = numerator / torch.sum(numerator, dim=1, keepdim=True)
#         return q