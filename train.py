import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ClustEncoderM import ClustEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd

torch.cuda.manual_seed(42)   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    "data_percentage": 0.1,       # 
    "lr": 0.1,               # learning rate
    "weight_decay": 0.0000001,  # weight decay
    "max_epoch": 50,
    "dims": [1024, 64, 64, 16],
    "n_cluster": 19,
    "alpha": 1
}


def main():
    model = ClustEncoder(config["dims"], activation=nn.GELU(), final_activation=None)
    print(model)
    # model = torch.compile(model)
    model = model.to(device)
    train(model)


def train(model):
    gene_data = torch.load('processed_data\\breast_g1\\gene_encode.pth')
    spatial_data = torch.load('processed_data\\breast_g1\\coord_encode.pth')
    img_data = torch.randn(4483, 384)
    cat_data = torch.cat((gene_data, spatial_data, img_data), dim=1).cuda() # [N, 1024]

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = F.kl_div


    loss_his =[]
    model.train()            
    for epoch in range(config["max_epoch"]):
    # for epoch in tqdm(range(config["max_epoch"])):
        z = model(cat_data) # [N, d]

        z_cpu = z.cpu().detach().numpy()
        cent = get_centre(z_cpu)


        q = soft_cluster(z, cent, config["alpha"])
        p = target_distribution(q)

        # print(q[0])
        # print(p[0])

        loss = criterion(q.log(), p, reduction='batchmean')
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_his.append(loss.item())
        
        entropy = -torch.sum(q * torch.log(q + 1e-6), dim=1).mean()
        print(f"[Epoch {epoch}] KL Loss: {loss.item():.4f}, Q entropy: {entropy.item():.4f}")
    
    print(q[0])
    print(p[0])    
    plot_graph(loss_his)
    torch.save(model.state_dict(), './chkpts/model.pth')

def get_centre(z):
    kmeans = KMeans(config["n_cluster"], n_init=30, random_state=42)

    pred = kmeans.fit_predict(z)

    centroid = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, requires_grad=False, device=device)
    return centroid
    
    # cent = Parameter(torch.Tensor(config["n_cluster"], z.shape[1]).cuda())
    # Group = pd.Series(pred, index=range(z.shape[0]), name="Group")
    # features = pd.DataFrame(z.detach().numpy(),index=np.arange(0,z.shape[0]))
    # merge_feature = pd.concat([features, Group], axis=1)
    # cluster_centers = np.asarray(merge_feature.groupby("Group").mean())
    # cluster_centers = torch.Tensor(cluster_centers)
    # cent.data.copy_(cluster_centers)

def loss_function(p, q):
    def kld(target, pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
    loss = kld(p, q)
    return loss

def soft_cluster( z, centroid, alpha):
    """
     Args:
        z: [N, d]
        centroid: [n_cluster, d]
    Returns:
        Tensor: [N, n_cluster]
    """
    dist_diff = (z.unsqueeze(1) - centroid)
    # print("dist_diff")
    # print(dist_diff[0])

    diff = torch.sum(dist_diff ** 2, 2) #[batch, 1, d] => [batch, n_cluster, d] => [batch, n_cluster]
    # print("diff")
    # print(diff[0])
    numerator = 1.0 / (1.0 + (diff / alpha))
    power = (alpha + 1.0) / 2
    numerator = numerator ** power
    q = numerator / torch.sum(numerator, dim=1, keepdim=True)
    return q # [batch, n_cluster]


def target_distribution(q):
    """
    Args:
        q: [N, n_cluster]
    Returns:
        Tensor: [N, n_cluster]
    """
    numerator = (q ** 2) / q.sum(0) # [batch, n_cluster] / [n_cluster]
    p = (numerator / numerator.sum(dim=1, keepdim=True)) # [n_cluster, batch] / [batch] => [batch, n_cluster]
    return p

def plot_graph(loss_his):
    # print(loss_his)
    plt.figure(figsize=(8, 4))
    plt.plot(loss_his, marker='o')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

main()