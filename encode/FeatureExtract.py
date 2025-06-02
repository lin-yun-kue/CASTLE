import torch
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer
from transformers import AutoModelForMaskedLM
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn

class GeneformerExtractor:
    def __init__(self, model_name = "ctheodoris/Geneformer"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TranscriptomeTokenizer(custom_attr_name_dict=None,
                                model_input_size=2048, special_token=False, collapse_gene_ids=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)


    def tokenize_data(self, from_dir, to_dir):
        self.tokenizer.tokenize_data(
            data_directory=from_dir,
            output_directory=to_dir,
            output_prefix = "tokenized",
            file_format = "h5ad"
        )

    def encode(self, to_dir):
        dataset = os.path.join(to_dir, "tokenized.dataset")
        tokenized_dataset = load_from_disk(dataset)
        input_ids_list = [torch.tensor(seq) for seq in tokenized_dataset["input_ids"]]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        loader = DataLoader(input_ids, batch_size = 256)
        all_hidden_states = []
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = batch.to(self.device)
                attention_mask = (batch!=0).long().to(self.device)
                outputs = self.model(input_ids=batch, attention_mask=attention_mask)
                all_hidden_states.append(outputs.hidden_states[-1].mean(1).cpu())
        hidden_states = torch.cat(all_hidden_states, dim=0)  # shape: (N, T, D)
        torch.save(hidden_states, os.path.join(to_dir, "gene_encode.pth"))

class SpatialExtractor2D:
    def __init__(self, dim_x = 64, dim_y = 64, max_x=10000, max_y = 10000, seed = 42):
        torch.manual_seed(seed)
        self.x_embed = nn.Embedding(max_x, dim_x)
        self.y_embed = nn.Embedding(max_y, dim_y)

    def encode(self, coords: torch.Tensor, to_dir):
        x = self.x_embed(coords[:, 0])
        y = self.y_embed(coords[:, 1])
        embeddings = torch.cat((x, y), dim = -1)
        print(f"writing embedding to {to_dir}")
        torch.save(embeddings, os.path.join(to_dir, "coord_encode.pth"))
