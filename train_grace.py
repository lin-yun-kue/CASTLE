import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib
import ClustEncoderM
importlib.reload(ClustEncoderM)
from ClustEncoderM import ClustAutoEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, silhouette_score
import os
from sklearn.decomposition import PCA
import umap
import ptsdae.model
importlib.reload(ptsdae.model)
import ptsdae.model as ae
from ptsdae.sdae import StackedDenoisingAutoEncoder
from Dataset import CellDataset, CellRawExpDataset, CellGeneformerDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from ptdec.dec import DEC
from ptdec.model import train, predict
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import community.community_louvain as community_louvain
from collections import defaultdict
# import os
# os.environ["WANDB_MODE"] = "disabled"
import wandb
import random
import datetime
import string

torch.cuda.manual_seed(42)   

cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    "pretrain_epoch": 200,
    "finetune_epoch": 50,
    "dec_epoch": 6000,
    "dims": [313, 70, 70, 500, 10],
    "batch_size": 128,
    "corrupt": 0.35,
    "n_cluster": 19,
    "alpha": 1,
    "plot": False,  # can be us                     ed to silence all plots
    "pretrain_lr": 0.001,
    "pretrain_momentum": 0.9,
    "pretrain_step_size": 500,
    "pretrain_gamma": 0.2,
    "num_workers": 4 if cuda else 0,
    "train_step_size": 100,
    "train_gamma": 1,
    "train_lr": 0.0001,
    "train_momentum": 0.9,
    "clusteralg": "louvain",
    "reduce_dim": "pca",
    "dec_lr": 0.005,
    "dec_step_size": 2000,
    "dec_gamma": 1,
    "stopping_delta":0.0000001
}

## data
data_dir = os.path.join("processed_data", "breast_g1")
gene_data = torch.load(os.path.join(data_dir, 'gene_encode.pth')).to(device)
spatial_data = torch.load(os.path.join(data_dir, 'coord_encode.pth')).to(device)
img_data = torch.load(os.path.join(data_dir, 'img_encode.pth')).to(device)
gene_raw_data = torch.load(os.path.join(data_dir, 'raw_expression.pth')).to(device)
ground_truth = torch.load(os.path.join(data_dir, 'ground_truth.pth')).to(device)
cat_data = torch.cat((gene_data, spatial_data, img_data), dim=1).to(device)  # [N, 1024]


def main():
    # test accuracy of various input
    # print("Ground truth: ")
    # eval_accuracy(ground_truth, ground_truth)
    _, pred = get_centre_pred(gene_raw_data, reduce_dim=config['reduce_dim'], clustering=config['clusteralg'], n_components=50)
    print("raw gene expression: ")
    eval_accuracy(pred, ground_truth)
    # _, pred = get_centre_pred(cat_data, reduce_dim=config['reduce_dim'], clustering=config['clusteralg'], n_components=50)
    # print("concatenated expression: ")
    # eval_accuracy(pred, ground_truth)
    # _, pred = get_centre_pred(gene_data,reduce_dim=config['reduce_dim'], clustering=config['clusteralg'], n_components=50)
    # print("geneformer encoded expression: ")
    # eval_accuracy(pred, ground_truth)
    # _, pred = get_centre_pred(img_data, reduce_dim=config['reduce_dim'], clustering=config['clusteralg'], n_components=50)
    # print("img encoded: ")
    # eval_accuracy(pred, ground_truth)
    # _, pred = get_centre_pred(spatial_data, reduce_dim=config['reduce_dim'], clustering=config['clusteralg'], n_components=50)
    # print("coord encoded: ")
    # eval_accuracy(pred, ground_truth)



    run_name = generateRunName()
    wandb.login(key="bee43af8ef4c8ec45c3bdfc2ce404e435d5f23cf")
    wandb.init(project="[AI539] Final Project", name=run_name, config=config)
    logger = wandb
    train_dataset = CellRawExpDataset()
    autoencoder = StackedDenoisingAutoEncoder(
        config["dims"], final_activation = None
    )
    autoencoder.to(device)
    print(autoencoder)
    print("model is launched on device:", device)
    print("pretraining stage-----")
    ae.pretrain(
        train_dataset,
        autoencoder,
        cuda = cuda,
        validation = None,
        epochs = config['pretrain_epoch'],
        batch_size = config['batch_size'],
        optimizer=lambda model: SGD(model.parameters(), lr=config['pretrain_lr'], momentum=config['pretrain_momentum']),
        scheduler=lambda x: StepLR(x, config['pretrain_step_size'], gamma=config['pretrain_gamma']),
        corruption=config['corrupt'],
        silent = False,
        num_workers= config['num_workers'],
        logger = logger,
    )
    print("training stage-----")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=config['train_lr'], momentum=config['train_momentum'])
    progress = tqdm(total=config['finetune_epoch'], desc="Training Progress")
    ae.train(
        train_dataset,
        autoencoder,
        cuda=cuda,
        validation=None,
        epochs= config['finetune_epoch'],
        batch_size=config['batch_size'],
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, config['train_step_size'], gamma=config['train_gamma']),
        corruption=config["corrupt"],
        num_workers=config['num_workers'],
        logger = logger,
        logger_desc = "train",
        progress_bar = progress,
    )
    progress.close()

    with torch.no_grad():
        z = autoencoder.encoder(gene_raw_data.to(torch.float32))

    # print(z)
    visualize_2d(z, ground_truth, logger=logger, descrip="SDAE")
    _, pred = get_centre_pred(z, clustering=config['clusteralg'])
    eval_accuracy(pred, ground_truth)

    print("DEC stage-----")
    model = DEC(cluster_number=config['n_cluster'], hidden_dimension=config["dims"][-1], encoder = autoencoder.encoder)
    dec_optimizer = SGD(params=model.parameters(), lr=config['dec_lr'], momentum=0.9)
    train(
        dataset= train_dataset,
        model=model,
        epochs=config['dec_epoch'],
        batch_size=config['batch_size'],
        optimizer=dec_optimizer,
        scheduler=StepLR(dec_optimizer, config['dec_step_size'], gamma=config['dec_gamma']),
        stopping_delta=config["stopping_delta"],
        cuda=cuda,
        logger = logger,
    )
    pred, true = predict(train_dataset, model, config["batch_size"], silent = True, return_actual=True, cuda = cuda)
    with torch.no_grad():
        z = model.encoder(gene_raw_data.to(torch.float32))
    visualize_2d(z, ground_truth, logger=logger, descrip="DEC")
    eval_accuracy(pred, true)


    # train self written version of autoencoder
    # model = ClustAutoEncoder(config["dims"], activation=nn.GELU(), final_encoder_activation=None, final_decoder_activation = None)
    # print(model)
    # model = model.to(device)
    #
    # train_auto_encoder(model)
    #
    # with torch.no_grad():
    #     z = model.encoder(cat_data)
    #
    # visualize_2d(z, ground_truth)
    # _, pred = get_centre_pred(z, reduce_dim='umap')
    # eval_accuracy(pred, ground_truth)
    #
    # pred_m = train(model.encoder)


    # eval_accuracy(pred_m)
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=auto_encode_config["lr"], weight_decay=auto_encode_config["weight_decay"])

    model.train()
    for epoch in range(auto_encode_config["max_epoch"]):
        out = model(cat_data)
        loss = criterion(out, cat_data)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), './chkpts/auto_model.pth')
    


def visualize_2d(z, ground_truth, logger, descrip, reduce_dim = "umap"):
    if not isinstance(z, np.ndarray):
        z = z.detach().cpu().numpy()
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.detach().cpu().numpy()
    vis_reducer = PCA(n_components=2) if reduce_dim != "umap" else umap.UMAP(n_components=2)
    z_2d = vis_reducer.fit_transform(z)
    fig = plt.figure()
    plt.figure()
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=ground_truth, cmap="tab10", s=5, alpha=0.6)
    plt.title(f"2D View after with Cluster Coloring")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Cluster")
    if not cuda:
        plt.show()
    logger.log({f"Viz/{descrip}": wandb.Image(fig)})


def get_centre_pred(z, reduce_dim = None, clustering = "kmeans" ,n_components = 10, n_neighbors = 15):
    if not isinstance(z, np.ndarray):
        z = z.detach().cpu().numpy()

    if reduce_dim == "pca":
        pca = PCA(n_components=n_components, random_state=42)
        z = pca.fit_transform(z)
    elif reduce_dim == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        z = reducer.fit_transform(z)
    else:
        print("Dimension not reduced")

    if clustering == "louvain":
        # create K-NN graph
        knn_graph = kneighbors_graph(z, n_neighbors=n_neighbors, include_self=False)
        G = nx.from_scipy_sparse_array(knn_graph)

        # run louvian
        partition = community_louvain.best_partition(G)
        pred = np.array([partition[i] for i in range(len(z))])

        # compute centroids
        cluster_dict = defaultdict(list)
        for idx, label in enumerate(pred):
            cluster_dict[label].append(z[idx])
        centroids_np = np.array([np.mean(cluster_dict[c], axis=0) for c in sorted(cluster_dict)])
        centroid = torch.tensor(centroids_np, dtype=torch.float32, device=device)
    else:
        kmeans = KMeans(config["n_cluster"], n_init=30, random_state=42)
        pred = kmeans.fit_predict(z)
        centroid = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, requires_grad=False, device=device)

    return centroid, pred
    


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

def eval_accuracy(pred, ground_truth):
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.detach().cpu().numpy()
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
    #
    # score = silhouette_score(gene_raw_data, pred, metric='euclidean')
    # print("Score:", score)

def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = ""+random_string+"_Multi30k"
  return run_name

main()