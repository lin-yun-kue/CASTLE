from torch.utils.data import Dataset, DataLoader
import torch
import scanpy as sc
import os
import pandas as pd
import importlib
import encode.FeatureExtract
importlib.reload(encode.FeatureExtract)
from encode.FeatureExtract import GeneformerExtractor
class STDataset(Dataset):
    def __init__(self, data_dir = "processed_data", samples = ["breast_g1"], sample_prop = 0.1,
                 ge_files = "gene_encode.pth", cell_images = "/HE.tif",
                 coords = "coord_encode.pth",
                 seed = 42, transform=None):
        self.sample_prop = sample_prop
        self.len = len(samples)
        self.ge_files = []
        self.coord_files = []
        for sample in samples:
            temp_ge = os.path.join(data_dir, sample, ge_files)
            self.ge_files.append(temp_ge)
            temp_coord = os.path.join(data_dir, sample, coords)
            self.coord_files.append(temp_coord)
        self.seed = seed

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
        expression = torch.load(self.ge_files[idx])
        cell_num = expression.shape[0]
        n_sample = int(cell_num * self.sample_prop)
        torch.manual_seed(self.seed)
        cell_idx = torch.randperm(cell_num)[:n_sample]
        expression = expression[cell_idx, :]

        # images
        images = torch.randn(n_sample, 3, 32, 32)

        # coordinates
        coords = torch.load(self.coord_files[idx])[cell_idx, :]

        return images, expression, coords

train_dataset = STDataset()
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory = True)
for batch in train_loader:
    images, expression, coord = batch
    print("Images shape:", images.shape)         # Expected: [1, N_cells, 3, 32, 32]
    print("Expression shape:", expression.shape) # Expected: [1, N_cells, g]
    print("Coord shape:", coord.shape)           # Expected: [1, N_cells, 2]

class CellDataset(Dataset):
    def __init__(self, data_dir="processed_data", samples="breast_g1",
                 ge_files="gene_encode_big.pth", cell_images="img_encode_big.pth",
                 coords="coord_encode_big.pth", ground_truth = "ground_truth_big.pth"):
        self.gene = torch.load(os.path.join(data_dir, samples, ge_files))
        self.coord = torch.load(os.path.join(data_dir, samples, coords))
        self.img = torch.load(os.path.join(data_dir, samples, cell_images))
        self.true = torch.load(os.path.join(data_dir, samples, ground_truth))
        self.len = self.coord.shape[0]

    def __len__(self):
        '''
        :return: [1] integer for number of tissue samples in dataset
        '''
        return self.len

    def __getitem__(self, idx):
        cat = torch.cat((self.gene[idx, :], self.coord[idx, :], self.img[idx, :]), dim=0)
        return cat, self.true[idx]


train_dataset = CellDataset()
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
for batch in train_loader:
    x, y = batch
    print("Concatenated Features shape:", x.shape)
    print("Truth shape:", y.shape)
    break