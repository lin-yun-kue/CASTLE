import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class DinoV2FeatureExtractor:
    def __init__(self, model_name: str = 'dinov2_vits14', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = "cpu"
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device).eval()

    def preprocess_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        [N, 3, H, W] 的 => tensor224x224

        Args:
            image_tensor (Tensor): [N, 3, H, W]，例如 [N, 3, 32, 32]

        Returns:
            Tensor: [N, 3, 224, 224]
        """
        # Resize to 224x224
        x = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False).to(self.device)

        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x.to(self.device)

    def extract_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        extract features from image

        Args:
            image_tensor (Tensor): [N, 3, 32, 32]

        Returns:
            Tensor: [N, D] dinov2_vits14:D=384; dinov2_vitb14:D=768
        """
        dataset = TensorDataset(image_tensor)
        loader = DataLoader(dataset, batch_size=10, num_workers=0)
        all_features = []
        with torch.no_grad():
            for (batch,) in tqdm(loader, desc="Extracting image features"):
                x = self.preprocess_tensor(batch)
                features = self.model(x)
                all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)
    

# image_tensor = torch.randn(4000, 3, 32, 32)  # [N, 3, 32, 32]
#
# extractor = DinoV2FeatureExtractor()
#
# features = extractor.extract_from_tensor(image_tensor)
#
# print(features.shape)
