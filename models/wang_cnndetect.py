import os
import torch
import torch.nn as nn
from torchvision.models import resnet50

class WangCNNDetector(nn.Module):
    """
    Sheng-Yu Wang et al. (CVPR 2020) CNNDetection Model:
    'CNN-generated images are surprisingly easy to spot... for now'
    Backbone: ResNet-50 trained on ProGAN/StyleGAN with blur & JPEG augmentation.
    Detects universal CNN generator artifacts and GAN structural footprints.
    """
    def __init__(self):
        super(WangCNNDetector, self).__init__()
        self.resnet = resnet50(weights=None)
        self.resnet.fc = nn.Linear(2048, 1)

    def forward(self, x):
        logit = self.resnet(x)
        prob = torch.sigmoid(logit)
        return prob, logit

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        model = cls()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.resnet.load_state_dict(clean_sd)
        model.to(device)
        model.eval()
        return model
