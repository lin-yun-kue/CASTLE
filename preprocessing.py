import os
import util
import importlib
import encode.FeatureExtract
importlib.reload(encode.FeatureExtract)
from encode.FeatureExtract import GeneformerExtractor, SpatialExtractor2D
importlib.reload(util)
import h5py
import torch
import encode.image_encode
importlib.reload(encode.image_encode)
from encode.image_encode import DinoV2FeatureExtractor
import pandas as pd

raw_dir = "data"
processed_dir = "processed_data"
samples = ["breast_g1"]
for sample in samples:
    ## gene expression--------------------------------------------------
    # need to rewrite h5ad file to a format that geneformer can tokenize
    selected_id = util.process_raw_expression(sample, thres = 200)

    from_dir = os.path.join(raw_dir, sample)
    to_dir = os.path.join(processed_dir, sample)

    current_encoded_gene  = torch.load(os.path.join(to_dir, "encoded_gene.pth"))
    if current_encoded_gene.shape[0] != len(selected_id):
        # initialize geneformer
        gene_extractor = GeneformerExtractor()
        print(gene_extractor.model)
        gene_extractor.tokenize_data(from_dir, to_dir) # create dataset in processed data
        gene_extractor.encode(to_dir) # save gene embedding to pth file



    # spatial coordinates
    coords = util.process_raw_coord(from_dir, selected_id)

    torch.save(coords, os.path.join(to_dir, "raw_coords.pth")) # raw
    spatial_extractor = SpatialExtractor2D()
    spatial_extractor.encode(coords, to_dir)

    # images
    selected_id_set = set(selected_id)
    selected_idx = []
    image_tensor = []
    with h5py.File(os.path.join(from_dir, 'cell_patch_sample.h5'), "r") as f:
        imgs = f["img"]
        barcodes = f["barcode"][:].astype(str)
        selected_idx = [i for i, b in enumerate(barcodes) if "".join(b.astype(str)) in selected_id_set]
        for i in selected_idx:
            arr = imgs[i]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()/255.0
            image_tensor.append(tensor)

    batch = torch.stack(image_tensor)
    extractor = DinoV2FeatureExtractor()
    features = extractor.extract_from_tensor(batch)
    torch.save(features, os.path.join(to_dir, "img_encode.pth"))

    ## ground truth label
    gt = pd.read_csv(os.path.join(from_dir, "clusters.csv"))
    gt["Barcode"] = gt["Barcode"].astype(str)
    gt = dict(zip(gt["Barcode"], gt["Cluster"]))
    selected_gt = [gt.get(id, -1) for id in selected_id]
    torch.save(torch.tensor(selected_gt, dtype = torch.int8), os.path.join(to_dir, "ground_truth.pth"))


with h5py.File(os.path.join(from_dir,"cell_feature_matrix.h5"), "r") as f:
    def print_all(name, obj):
        print(name, "->", obj)

    f.visititems(print_all)
    barcodes_real = [b.decode() for b in f["matrix"]["barcodes"][:]]



