from torch.utils.data import Dataset, DataLoader
import torch
import scanpy as sc
import os
import pandas as pd
class STDataset(Dataset):
    def __init__(self, data_dir = "data", samples = ["breast_g1"], sample_prop = 0.1,
                 ge_files = "cell_feature_matrix.h5", cell_images = "/HE.tif",
                 coords = "cells.csv.gz",
                 transform=None):
        '''
        Initializing dataset, doesn't return anything.
        :param data_dir: path for datafolder that contains one folder for each tissue sample
        :param sample_prop: the proportion of cells in each tissue sample to use
        :param ge_files: path for gene expression data within tissue sample folder
        :param cell_images: path for cropped cell image patches  within tissue sample folder
        :param transform: cell image transformation
        '''
        self.sample_prop = sample_prop
        self.len = len(samples)
        self.ge_files = []
        self.coord_files = []
        for sample in samples:
            temp_ge = os.path.join(data_dir, sample, ge_files)
            self.ge_files.append(temp_ge)
            temp_coord = os.path.join(data_dir, sample, coords)
            self.coord_files.append(temp_coord)

    def __len__(self):
        '''
        :return: [1] integer for number of tissue samples in dataset
        '''
        return self.len

    def __getitem__(self, idx):
        '''
        :param idx: default input for dataloader, no need to manually input
        :return images: [N_cells*sample_prop, 3, 32, 32]
        :return expression: [N_cells*sample_prop, g]  Note, N cells represent the number of cells in each tissue sample,
            g is the number of genes in each tissue sample. Both of them can differ for each tissue sample
        :return coord: [N_cells*sample_prop, 2]
        :return gene_names: [g] vector of the genes stored in expression
        '''

        # gene expression
        ge_file = sc.read_10x_h5(self.ge_files[idx])
        cell_num = ge_file.shape[0]
        n_sample = int(cell_num * self.sample_prop)
        cell_idx = torch.randperm(cell_num)[:n_sample]
        expression = torch.tensor(ge_file.X.toarray(), dtype=torch.int8)[cell_idx, :]

        # images
        images = torch.randn(n_sample, 3, 32, 32)

        # coordinates
        coord_table = pd.read_csv(self.coord_files[idx])
        coords = torch.tensor(coord_table[["x_centroid", "y_centroid"]].values, dtype=torch.float)[cell_idx, :]

        gene_names = ge_file.var_names.tolist()

        return images, expression, coords, gene_names

train_dataset = STDataset()
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory = True)
for batch in train_loader:
    images, expression, coord, gene_names = batch
    print("Images shape:", images.shape)         # Expected: [1, N_cells, 3, 32, 32]
    print("Expression shape:", expression.shape) # Expected: [1, N_cells, g]
    print("Coord shape:", coord.shape)           # Expected: [1, N_cells, 2]
    print("Gene names shape:", len(gene_names)) # Expected: should error or be inconsistent as gene_names is a list of strings
    print(gene_names[:10])
    break