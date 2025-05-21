import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ClustEncoderM import ClustEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.cluster import KMeans



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    "data_percentage": 0.1,       # 
    "lr":0.0003,               # learning rate
    "weight_decay":0.0000001,  # weight decay
    "max_epoch":100,
    # "opt": "admin",             # admin | sgd
    # "init": "k-mean",
    "dims": [64, 32, 16, 8],
    "n_cluster": 5,
    "alpha": 1
}


def main():

    model = ClustEncoder(config["dims"], alpha=config["alpha"])
    model = torch.compile(model)
    model.to(device)
    train(model)


def train(model, dataloader=None):


    optimizer = optim.Adam(model.parameters(),lr=config["lr"], weight_decay=config["weight_decay"])
    # scheduler = CosineAnnealingLR(optimizer, T_max = config["max_epoch"])

    criterion = F.kl_div


    # get centroid
    kmeans = KMeans(config["n_cluster"], n_init=30, random_state=42)
    data = dataloader.data #todo 
    data = data.to(device)
    encoded_data = model.encoder(data).detach().cpu()
    pred = kmeans.fit_predict(encoded_data)
    centroid = torch.tensor(kmeans.cluster_centers_, requires_grad=True).cuda()



    for epoch in tqdm(range(config["max_epoch"])):

        model.train()

        #todo
        for x in dataloader:

            x = x.to(device)
            
            z = model(x) # [batch, d]
            q = soft_cluster(z, centroid, config["alpha"])
            p = target_distribution(q)
            loss = criterion(q.log(), p)

            loss.backward()
            optimizer.step()

            print(f"Loss/train: {loss}")
    
    torch.save(model.state_dict(), './chkpts/model.pth')


def soft_cluster( z, centroid, alpha):
    # z: [batch, d]
    # centroid: [n_cluster, d]
    diff = torch.sum((z.unsqueeze(1) - centroid) ** 2, 2) #[batch, 1, d] => [batch, n_cluster, d] => [batch, n_cluster]
    numerator = 1.0 / (1.0 + (diff / alpha))
    power = (alpha + 1.0) / 2
    numerator = numerator ** power
    q = numerator / torch.sum(numerator, dim=1, keepdim=True)
    return q # [batch, n_cluster]


def target_distribution(q):
    numerator = (q ** 2) / torch.sum(q, 0) # [batch, n_cluster] / [n_cluster]
    p = (numerator.t() / torch.sum(numerator, 1)).t() # [n_cluster, batch] / [batch] => [batch, n_cluster]
    return p






























main()