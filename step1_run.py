import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib
import ClustEncoderM

importlib.reload(ClustEncoderM)
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, silhouette_score
import os
from sklearn.decomposition import PCA
import umap
import ptsdae.model
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import community.community_louvain as community_louvain
from collections import defaultdict
import random
import datetime
import string
from collections import Counter

# torch.cuda.manual_seed(42)

cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    "n_component": [8, 16, 32, 64, 128],
    "clustering": ["kmeans", "louvain"],
    "dim_reduction": ["pca", "umap"],
    "n_cluster": 19
}

import warnings

# Suppress sklearn's force_all_finite deprecation warning
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8",
    category=FutureWarning
)

# Suppress UMAP's n_jobs override warning
# warnings.filterwarnings(
#     "ignore",
#     message="n_jobs value 1 overridden to 1 by setting random_state.*",
#     category=UserWarning
# )

## data
data_dir = os.path.join("processed_data", "breast_g1")
gene_data = torch.load(os.path.join(data_dir, 'gene_encode_small.pth')).to(device)
spatial_data = torch.load(os.path.join(data_dir, 'coord_encode_small.pth')).to(device)
img_data = torch.load(os.path.join(data_dir, 'img_encode_small.pth')).to(device)
# spatial_data_2 = torch.load(os.path.join(data_dir, 'coord_encode_2.pth')).to(device)
# img_data_2 = torch.load(os.path.join(data_dir, 'img_encode_2.pth')).to(device)
gene_raw_data = torch.load(os.path.join(data_dir, 'raw_expression_small.pth')).to(device)
ground_truth = torch.load(os.path.join(data_dir, 'ground_truth_small.pth')).to(device)


def main():
    columns = ["n_component", "clustering", "dim_reduction", "data_type", "acc", "score", "entropy"]
    results_df = pd.DataFrame(columns=columns)
    for n in config["n_component"]:
        for c in config["clustering"]:
            for d in config["dim_reduction"]:
                counter = 1
                for data in [gene_raw_data, gene_data, spatial_data, img_data]:
                    _, pred, features = get_centre_pred(data, reduce_dim=d, clustering=c, n_components=n)
                    acc, score, entropy = eval_accuracy(pred, ground_truth, features)
                    results_df.loc[len(results_df.index)] = [n, c, d, counter,acc, score, entropy]
                    counter += 1
                    print(f"{n}, {c}, {d}, {counter}, {acc}, {score}, {entropy}")
    print(results_df)
    results_df.to_csv("results_df.csv")


def get_centre_pred(z, reduce_dim=None, clustering="kmeans", n_components=10, n_neighbors=15):
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

    return centroid, pred, z


def eval_accuracy(pred, ground_truth, features = None):
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

    score = silhouette_score(features, pred, metric='euclidean')
    entropy = label_entropy(y_pred_mapped, y_true)


    return acc, score, entropy


def generateRunName():
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    now = datetime.datetime.now()
    run_name = "" + random_string + "_Multi30k"
    return run_name


def label_entropy(pred, true):
    N = len(true)
    cluster_labels = np.unique(pred)
    total_entropy = 0.0

    for cluster in cluster_labels:
        idxs = np.where(pred == cluster)[0]
        true_labels_in_cluster = ground_truth[idxs]
        cluster_size = len(idxs)

        # compute p_{i|k}
        label_counts = Counter(true_labels_in_cluster)
        probs = np.array([count / cluster_size for count in label_counts.values()])

        # entropy of this cluster
        cluster_entropy = -np.sum(probs * np.log2(probs + 1e-10))  # avoid log(0)
        weight = cluster_size / N

        total_entropy += weight * cluster_entropy

    return total_entropy

main()