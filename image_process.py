import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import util
import h5py
from PIL import Image

sample = "breast_g1"
data_dir = os.path.join("data", sample)

# based on experiment.xenium parameter
pixel = 0.2125

# read in cell coordinates & affline matrix-------------------------------
coords = pd.read_csv(os.path.join(data_dir, "cells.csv.gz"))  # or use "cells.parquet" with pd.read_parquet
align = pd.read_csv(os.path.join(data_dir, "align.csv"), header = None)

# convert coordinates to pixels ------------------------------------------
# extract only x, y coordinate
coords_converted = coords[["x_centroid", "y_centroid"]].to_numpy()

# convert from micrometer to pixel
coords_converted = coords_converted / pixel

# add homogeneous coordinate
coords_converted = np.c_[coords_converted, np.ones(coords_converted.shape[0])]
coords_converted =  coords_converted.transpose()

# apply affline transformation
coords_converted = np.matmul(np.linalg.inv(align.to_numpy()), coords_converted)
final_coords  = coords_converted.T[:, :2]


# cut patches ---------------------------------------------------------------
util.process_and_visualize_image(
    data_dir, data_dir, final_coords, target_patch_size=32, barcodes=coords["cell_id"])

with h5py.File(os.path.join(data_dir,"cell_patch_sample.h5"), "r") as f:
    imgs = f["img"]  # assumes the dataset is named 'img'
    print(f"Total images: {len(imgs)}")
    print(list(f.keys()))
    for i in range(10):
        arr = imgs[i]
        img = Image.fromarray(arr)

        # Show using matplotlib
        plt.imshow(img)
        plt.title(f"Image {i}")
        plt.axis("off")
        plt.show()

