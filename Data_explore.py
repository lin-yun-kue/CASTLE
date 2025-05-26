import json
import pprint
import os
data_folder = "data/breast_g1"

### gene names
with open("data/breast_g1/gene_panel.json") as f:
    gene_panel = json.load(f)
# Explore the structure of the json file
# print(gene_panel.keys())
# pprint.pprint(gene_panel["payload"], depth=2)

gene_names = [gene["type"]["data"]["name"] for gene in gene_panel["payload"]["targets"]]
print(gene_names)

### cell coordinates
import pandas as pd
align = pd.read_csv(os.path.join(data_folder, "align.csv"))
# Load the CSV
cells = pd.read_csv(os.path.join(data_folder, "cells.csv.gz"))  # or use "cells.parquet" with pd.read_parquet
coords = cells[["x_centroid", "y_centroid"]]

# Preview columns
print(cells.columns)
cells.head()

### gene expression matrix
import scanpy as sc
import torch
adata = sc.read_10x_h5(os.path.join(data_folder, "cell_feature_matrix.h5"))
print(adata)
adata.X[:3].todense() # we do need to care whether we need to normalize the gene expression data
adata.var_names[:10] ## can be used as column names too
gene_expression = torch.tensor(adata.X.toarray(), dtype=torch.float32)
gene_expression.shape

import tifffile as tiff

image = tiff.imread(os.path.join(data_folder,"HE.tif"))

import util
img = os.path.join(data_folder)
patch_file = os.path.join(data_folder,"image_patch")
coords = cells[["x_centroid", "y_centroid"]]
coord = coords.values.tolist()
cell_id = cells["cell_id"].values.tolist()

# Load the image and transpose it to the correct format
print("Loading imgs ...")
intensity_image = tiff.imread(os.path.join(img, "HE.tif"))

print("Patching: create image dataset (X) ...")
# Path for the .h5 image dataset
h5_path = os.path.join(patch_file)

import importlib
import util
importlib.reload(util)
# Create the patcher object to extract patches (localized square sub-region of an image) from an image at specified coordinates.
patcher = util.Patcher(
    image=intensity_image,
    coords=coord,
    patch_size_target=32
)

# Build and Save patches to an HDF5 file
patcher.to_h5(h5_path, extra_assets={'barcode': cell_id})
patcher.view_coord_points()
# Delete variables that are no longer used
del intensity_image, patcher
gc.collect()


import h5py
import matplotlib.pyplot as plt
# Open the file in read mode
with h5py.File("data/breast_g1/image_patch.h5", "r") as f:
    # List all top-level keys (groups or datasets)
    print("Keys in file:", list(f.keys()))
    print("img shape:", f["img"].shape)
    print("dtype:", f["img"].dtype)
    imgs = f["img"]
    for i in range(5):
        img = imgs[i]

        # Convert to H x W x C if needed
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # assume channel-first
            img = img.transpose(1, 2, 0)

        # If grayscale, squeeze to (H, W)
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        plt.imshow(img.astype("uint8") if img.dtype != 'uint8' else img)
        plt.title(f"Image {i}")
        plt.axis("off")
        plt.show()
        print(f["coords"][i])
    # Optionally, explore subgroups recursively
    # def visit_all(name):
    #     print(name)
    # f.visit(visit_all)

import numpy as np
plt.imshow(image)
plt.show()
coor_array = np.array(coord)
microns_per_pixel =  0.924
coor_array = coor_array / microns_per_pixel
plt.scatter(coor_array[:, 0], coor_array[:, 1], s =1, c='red')

with tiff.TiffFile("data/breast_g1/HE.tif") as tif:
    px_size_x = tif.pages[0].tags["XResolution"].value
    px_size_y = tif.pages[0].tags["YResolution"].value
    print("Pixel size:", px_size_x, px_size_y)

align = pd.read_csv(os.path.join(data_folder, "align.csv"), header=None)
affine = np.array(align)
coords_um = np.array(coords)
coords_h = np.concatenate([coords_um, np.ones((coords_um.shape[0], 1))], axis=1)
coords_px = coords_h @ affine
coords_px = coords_px[:, :2]