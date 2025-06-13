import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.path.join("processed_data", "breast_g1")
gene_data = torch.load(os.path.join(data_dir, 'gene_encode.pth'))
spatial_data = torch.load(os.path.join(data_dir, 'coord_encode.pth'))
img_data = torch.load(os.path.join(data_dir, 'img_encode.pth'))
gene_raw_data = torch.load(os.path.join(data_dir, 'raw_expression.pth'))
ground_truth = torch.load(os.path.join(data_dir, 'ground_truth.pth'))
cat_data = torch.cat((gene_data, spatial_data, img_data), dim=1)





