import os
import util
import importlib
import encode.FeatureExtract
importlib.reload(encode.FeatureExtract)
from encode.FeatureExtract import GeneformerExtractor, SpatialExtractor2D
importlib.reload(util)


raw_dir = "data"
processed_dir = "processed_data"
samples = ["breast_g1"]
for sample in samples:
    ## gene expression--------------------------------------------------
    # need to rewrite h5ad file to a format that geneformer can tokenize
    selected_id = util.process_raw_expression(sample, thres = 500)

    from_dir = os.path.join(raw_dir, sample)
    to_dir = os.path.join(processed_dir, sample)

    # initialize geneformer
    gene_extractor = GeneformerExtractor()
    gene_extractor.tokenize_data(from_dir, to_dir) # create dataset in processed data
    gene_extractor.encode(to_dir) # save gene embedding to pth file

    # spatial coordinates
    coords = util.process_raw_coord(from_dir, selected_id)
    spatial_extractor = SpatialExtractor2D()
    spatial_extractor.encode(coords, to_dir)





