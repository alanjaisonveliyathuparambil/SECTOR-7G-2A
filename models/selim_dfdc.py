import os
import torch
import torch.nn as nn
import timm

class SelimDFDCClassifier(nn.Module):
    """
    Selim Seferbekov's 1st Place Kaggle DFDC Deepfake Detection Solution Architecture.
    Backbone: EfficientNet-B7 Noisy Student (tf_efficientnet_b7_ns)
    Head: Linear(2560, 1) with Sigmoid activation.
    """
    def __init__(self, model_name="tf_efficientnet_b7_ns", drop_rate=0.2):
        super(SelimDFDCClassifier, self).__init__()
        # timm backbone without original 1000-class classifier
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_rate=drop_rate
        )
        self.fc = nn.Linear(2560, 1)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.fc(features)
        prob = torch.sigmoid(logits)
        return prob, logits

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        model = cls()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # Clean potential DataParallel prefix
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # Remove unused classifier weights from backbone if present
        clean_sd = {k: v for k, v in clean_sd.items() if not k.startswith("encoder.classifier")}
        model.load_state_dict(clean_sd, strict=False)
        model.to(device)
        model.eval()
        return model
