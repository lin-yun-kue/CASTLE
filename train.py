import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib
import ClustEncoderM
importlib.reload(ClustEncoderM)
from ClustEncoderM import ClustEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os
from sklearn.decomposition import PCA
import umap

torch.cuda.manual_seed(42)   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = {
    "data_percentage": 0.1,       #
    "lr": 0.1,               # learning rate
    "weight_decay": 0.0000001,  # weight decay
    "max_epoch": 100,
    "dims": [1024, 64, 16],
    "n_cluster": 19,
    "alpha": 1,
    "plot": False # can be used to silence all plots
}


def main():
    model = ClustEncoder(config["dims"], activation=nn.GELU(), final_encoder_activation=None)
    print(model)
    # model = torch.compile(model)
    model = model.to(device)
    # pred_m, pred_c, pred_g, pred_r = train(model)
    pred_m = train(model)

    
    eval_accuracy(pred_m)
    # eval_accuracy(pred_c)
    # eval_accuracy(pred_g)
    # eval_accuracy(pred_r)

def plot_class_graph(pred):
    unique, counts = np.unique(pred, return_counts=True)
    
    plt.bar(unique, counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Prediction Histogram')
    plt.show()

def train_auto_encoder(model):
    print("")

def train(model):
    data_dir = os.path.join("processed_data", "breast_g1")
    gene_data = torch.load(os.path.join(data_dir,'gene_encode.pth'))
    spatial_data = torch.load(os.path.join(data_dir, 'coord_encode.pth'))
    img_data = torch.load(os.path.join(data_dir, 'img_encode.pth'))
    gene_raw_data = torch.load(os.path.join(data_dir, 'raw_expression.pth'))
    data_dir = os.path.join("processed_data", "breast_g1")
    ground_truth = torch.load(os.path.join(data_dir, 'ground_truth.pth'))
    cat_data = torch.cat((gene_data, spatial_data, img_data), dim=1).to(device) # [N, 1024]

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = F.kl_div


    loss_his =[]
    model.train()            
    for epoch in range(config["max_epoch"]):
    # for epoch in tqdm(range(config["max_epoch"])):
        z = model(cat_data) # [N, d]

        z_cpu = z.cpu().detach().numpy()
        cent, _ = get_centre_pred(z_cpu, ground_truth)


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
    if config["plot"]:
        plot_graph(loss_his)
    torch.save(model.state_dict(), './chkpts/model.pth')

    ## final cluster output
    with torch.no_grad():
        z = model(cat_data)
        z_cpu = z.cpu().detach().numpy()
        z_cent, _ = get_centre_pred(z_cpu, ground_truth)
        pred_m = soft_cluster(z, z_cent, config["alpha"]).cpu().detach()
        pred_m = np.argmax(pred_m, axis=1)

    # _, pred_c = get_centre_pred(cat_data.cpu().detach().numpy(), ground_truth, reduce_dim="umap")
    # _, pred_g = get_centre_pred(gene_data.cpu().detach().numpy(), ground_truth, reduce_dim = "umap")
    # _, gene_r = get_centre_pred(gene_raw_data.cpu().detach().numpy(), ground_truth, reduce_dim = "umap")

    return pred_m
    # return pred_m, pred_c, pred_g, gene_r


def get_centre_pred(z, ground_truth, reduce_dim = None, n_components = 10):
    if reduce_dim == "pca":
        pca = PCA(n_components=n_components, random_state=42)
        z = pca.fit_transform(z)
    elif reduce_dim == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        z = reducer.fit_transform(z)
    else:
        print("Dimension not reduced")

    ground_truth = ground_truth.cpu().detach().numpy()
    kmeans = KMeans(config["n_cluster"], n_init=30, random_state=42)

    pred = kmeans.fit_predict(z)

    centroid = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, requires_grad=False, device=device)
    if config['plot'] and n_components > 2:
        vis_reducer = PCA(n_components=2) if reduce_dim != "umap" else umap.UMAP(n_components=2)
        z_2d = vis_reducer.fit_transform(z)
        plt.figure()
        plt.scatter(z_2d[:, 0], z_2d[:, 1], c=ground_truth, cmap="tab10", s=5, alpha=0.6)
        plt.title(f"2D View after with Cluster Coloring")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(label="Cluster")
        plt.show()

    return centroid, pred
    
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

def eval_accuracy(pred):
    data_dir = os.path.join("processed_data", "breast_g1")
    ground_truth = torch.load(os.path.join(data_dir,'ground_truth.pth'))
    ground_truth = ground_truth.cpu().detach().numpy()
    y_true = np.asarray(ground_truth) - 1
    y_pred = np.asarray(pred)
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    majority_map = (
        df.groupby('pred')['true']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    y_pred_mapped = df['pred'].map(majority_map).to_numpy()
    acc = accuracy_score(y_true, y_pred_mapped)
    # cm = confusion_matrix(y_true, y_pred_mapped)
    # acc = accuracy_score(y_true, y_pred)
    if config['plot']:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
        disp.plot(cmap = "Blues")
        plt.title("Confusion Matrix")
        plt.show()
    print("Accuracy:", acc)
    # print("confusion:", cm)


main()