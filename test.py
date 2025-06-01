import torch
import torch.nn as nn
from sklearn.cluster import KMeans

gene_data = torch.load('processed_data\\breast_g1\\gene_encode.pth')
spatial_data = torch.load('processed_data\\breast_g1\\coord_encode.pth')
img_data = torch.randn(4483, 384)
cat_data = torch.cat((gene_data, spatial_data, img_data), dim=1).detach() # [5, 1024]


kmeans = KMeans(19, n_init=30, random_state=42)
pred = kmeans.fit(cat_data)

d = kmeans.transform(cat_data)

dists = torch.tensor(d)
q = 1.0 / (1.0 + (dists**2))  
q = q / q.sum(dim=1, keepdim=True)


print(q[0])