import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ClustEncoderM import ClustEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.cluster import KMeans


torch.cuda.manual_seed(42)   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    "data_percentage": 0.1,       # 
    "lr": 0.0003,               # learning rate
    "weight_decay": 0.0000001,  # weight decay
    "max_epoch": 10,
    # "opt": "admin",             # admin | sgd
    # "init": "k-mean",
    "dims": [1024, 256, 64],
    "n_cluster": 5,
    "alpha": 1
}


def main():
    model = ClustEncoder(config["dims"], activation=nn.Tanh(), final_activation=nn.Tanh())
    print(model)
    # model = torch.compile(model)
    model = model.to(device)
    train(model)


def train(model):


    optimizer = optim.Adam(model.parameters(),lr=config["lr"], weight_decay=config["weight_decay"])
    # scheduler = CosineAnnealingLR(optimizer, T_max = config["max_epoch"])

    criterion = F.kl_div


    # get centroid
    kmeans = KMeans(config["n_cluster"], n_init=30, random_state=42)

    gene_data = torch.load('processed_data\\breast_g1\\gene_encode.pth')
    spatial_data = torch.load('processed_data\\breast_g1\\coord_encode.pth')
    img_data = torch.randn(4483, 384)
    combined_data = torch.cat((gene_data, spatial_data, img_data), dim=1).detach() # [5, 1024]
    # print(combined_data[0])
    # print(combined_data.shape)

    pred = kmeans.fit_predict(combined_data)
    
    centroid = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, requires_grad=False, device=device)
    model.eval()
    centroid_z = model(centroid).detach()
    # print("centroid Z")
    # print(centroid_z[0])

    combined_data = combined_data.cuda()


    model.train()            
    for epoch in tqdm(range(config["max_epoch"])):

        z = model(combined_data) # [N, d]
        # print("Z")
        # print(z[0])
        q = soft_cluster(z, centroid_z, config["alpha"])
        # print("Q")
        # print(q[1])
        p = target_distribution(q)
        # print("P")
        # print(p[0])

        loss = criterion(q.log(), p)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss/train: {loss}")
    
    torch.save(model.state_dict(), './chkpts/model.pth')


def soft_cluster( z, centroid, alpha):
    # z: [batch, d]
    # centroid: [n_cluster, d]
    diff = torch.sum((z.unsqueeze(1) - centroid) ** 2, 2) #[batch, 1, d] => [batch, n_cluster, d] => [batch, n_cluster]
    print("diff")
    print(diff[0])
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