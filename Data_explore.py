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
import os
data_folder = "data/breast_g1"
adata = sc.read_10x_h5(os.path.join(data_folder, "cell_feature_matrix.h5"))
adata.var["gene_ids"]
print(adata)
adata.X[:3].todense() # we do need to care whether we need to normalize the gene expression data
adata.var_names[:10] ## can be used as column names too
type(adata.var_names)
pd.Index(adata.var_names).astype(str).tolist()
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

# check the token dictionary
import pickle
import mygene
with open("encode/token_dictionary_gc30M.pkl", "rb") as f:
    toekn_dict = pickle.load(f)
len(toekn_dict.keys())
toekn_dict["<cls>"]
"<cls>" in toekn_dict
toekn_dict.keys()
toekn_dict["ACTA2"]
mg = mygene.MyGeneInfo()
symbols = ["ACTA2"]
result = mg.querymany(symbols, scopes = "symbol", fields = "ensembl.gene", species = "human")
result["ensembl"]
max_token_id = max(toekn_dict.values())
next_token_id = max_token_id + 1
# # Add <cls> and <eos> if not already present
# if "<cls>" not in toekn_dict:
#     toekn_dict["<cls>"] = next_token_id
#     next_token_id += 1
#
# if "<eos>" not in toekn_dict:
#     toekn_dict["<eos>"] = next_token_id
#
# # Save the patched dictionary
# with open("encode/token_dictionary_gc30M_patched.pkl", "wb") as f:
#     pickle.dump(toekn_dict, f)


### extract with geneformer
import scanpy as sc
import numpy as np
data_folder = "data/breast_g1"
adata = sc.read_10x_h5(os.path.join(data_folder, "cell_feature_matrix.h5"))
adata.var.rename(columns={"gene_ids": "ensembl_id"}, inplace=True)
adata.obs["n_counts"] = adata.X.sum(axis=1).A1 if hasattr(adata.X, "A1") else np.array(adata.X.sum(axis=1)).flatten().astype(int)
min(adata.obs["n_counts"].values)
adata.obs["filter_pass"] = 1
adata.write("data/breast_g1/preprocessed.h5ad")

df = pd.read_parquet("data/breast_g1/transcripts.parquet")


from geneformer import TranscriptomeTokenizer

tokenizer = TranscriptomeTokenizer(
    custom_attr_name_dict=None,       # or {"cell_type": "cell_type"} if needed
    model_input_size=2048,            # or 4096 for 95M
    special_token=False,              # True for 95M
    collapse_gene_ids=True
)

tokenizer.tokenize_data(
    data_directory="data/breast_g1",                # current directory
    output_directory="processed_data/breast_g1",
    output_prefix="tokenized",
    file_format="h5ad"
)

from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
tokenized_dataset = load_from_disk("processed_data/breast_g1/tokenized.dataset")
input_ids_list = [torch.tensor(seq) for seq in tokenized_dataset["input_ids"]]
input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
attention_mask = (input_ids != 0).long()  # padding mask

# Move to device
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Use the 30M or 95M model from Hugging Face
model_name = "ctheodoris/Geneformer"  # or "ctheodoris/Geneformer-95M"

model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

tokenizer = AutoTokenizer.from_pretrained(model_name)


## explore H&E image again
pixel = 0.2125
import pandas as pd
import os
import numpy as np
data_folder = "data/breast_g1"
align = pd.read_csv(os.path.join(data_folder, "align.csv"), header = None)
# Load the CSV
cells = pd.read_csv(os.path.join(data_folder, "cells.csv.gz"))  # or use "cells.parquet" with pd.read_parquet
coords = cells[["x_centroid", "y_centroid"]]
coords = coords/pixel
coords
coords = coords.to_numpy()
coords = np.c_[coords, np.ones(coords.shape[0])]
coords = coords.transpose()
converted = np.matmul(np.linalg.inv(align.to_numpy()), coords)
converted.T[:10, : ]

##
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# image = tiff.imread(os.path.join(data_folder,"HE.tif"))
sample_coords = converted.transpose()[:10, :1]
cells[["x_centroid", "y_centroid"]][:10]
with h5py.File("data/breast_g1/cell_patch_sample.h5", "r") as f:
    imgs = f["img"]  # assumes the dataset is named 'img'
    print(f"Total images: {len(imgs)}")

    for i in range(len(imgs)):
        arr = imgs[i]
        img = Image.fromarray(arr)

        # Show using matplotlib
        plt.imshow(img)
        plt.title(f"Image {i}")
        plt.axis("off")
        plt.show()

data_dir = os.path.join("processed_data", "breast_g1")
gene_data = torch.load(os.path.join(data_dir, 'gene_encode.pth'))
spatial_data = torch.load(os.path.join(data_dir, 'coord_encode.pth'))
img_data = torch.load(os.path.join(data_dir, 'img_encode.pth'))
gene_raw_data = torch.load(os.path.join(data_dir, 'raw_expression.pth'))

import matplotlib.pyplot as plt
def plot_hist(filename, title, sample_size=10000):
    data = torch.load(os.path.join(data_dir, filename)).flatten()
    if data.numel() > sample_size:
        data = data[torch.randperm(data.numel())[:sample_size]]
    plt.hist(data.cpu().detach().numpy(), bins=100)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

plot_hist('gene_encode.pth', 'Gene Encoded Data')
plot_hist('coord_encode.pth', 'Spatial Encoded Data')
plot_hist('img_encode.pth', 'Image Encoded Data')
plot_hist('raw_expression.pth', 'Raw Gene Expression Data')

import torch
a = torch.load("processed_data/breast_g1/gene_encode.pth")
