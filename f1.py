import os
from typing import Dict, Callable, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision import transforms
from torchvision.models.resnet import resnet152, ResNet152_Weights

from src.dataset import StomachCancerDataset
from src.dataloader import build_loader


CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
LABEL_MAP = {label: i for i, label in enumerate(CLASSES)}


def build_fc_decrescente_linear(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
    )

def build_fc_decrescente_sigmoid(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Sigmoid(),
        nn.Linear(1024, 512),
        nn.Sigmoid(),
        nn.Linear(512, 64),
        nn.Sigmoid(),
        nn.Linear(64, 8),
    )

def build_fc_1024_512_bn_dropout(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 8),
    )

def build_fc_tanh(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.Tanh(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.Tanh(),
        nn.Dropout(0.3),
        nn.Linear(128, 8),
    )

def build_fc_gelu(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 2048),
        nn.GELU(),
        nn.Linear(2048, 512),
        nn.GELU(),
        nn.Linear(512, 8),
    )

def build_fc_silu(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.SiLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.SiLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 8),
    )


ARCHITECTURES: Dict[str, Callable[[int], nn.Sequential]] = {
    "fc_2048-64_relu": build_fc_decrescente_linear,
    "fc_1024-64-sigmoid": build_fc_decrescente_sigmoid,
    "fc_1024-256_relu": build_fc_1024_512_bn_dropout,
    "fc_512-128_tanh": build_fc_tanh,
    "fc_2048-512_gelu": build_fc_gelu,
    "fc_1024-256_silu": build_fc_silu,
}


def load_dataset(csv_file: str):
    df = pd.read_csv(csv_file)
    images = df["path"].values
    labels = df["label"].map(LABEL_MAP).values
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = StomachCancerDataset(images, labels, transform)
    return build_loader(dataset, batch_size=64, num_workers=3)


def get_model(arch: str, device: torch.device) -> nn.Module:
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = ARCHITECTURES[arch](model.fc.in_features)
    model.to(device)
    return model


@torch.inference_mode()
def predict(model: nn.Module, loader) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    device = next(model.parameters()).device

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

    return np.array(y_true), np.array(y_pred)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro")


def main():
    device = torch.device("cuda")
    splits = {
        "Treino": "train.csv",
        "Validação": "validation.csv",
        "Teste": "test.csv",
    }

    results = []

    for arch in ARCHITECTURES:
        model_path = os.path.join("results", "ResNet", arch, "model.pth")
        if not os.path.exists(model_path):
            print(f"Modelo não encontrado: {arch}")
            continue

        model = get_model(arch, device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        for split_name, csv_file in splits.items():
            loader = load_dataset(csv_file)
            y_true, y_pred = predict(model, loader)
            f1_macro = compute_f1(y_true, y_pred)
            results.append({
                "architecture": arch,
                "split": split_name,
                "f1_macro": f1_macro
            })


    df = pd.DataFrame(results)
    df.to_csv("f1_scores.csv", index=False)
    print(df.head())


if __name__ == "__main__":
    main()
