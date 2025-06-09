import os
from typing import Dict, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
    "fc_decrescente_linear": build_fc_decrescente_linear,
    "fc_decrescente_sigmoid": build_fc_decrescente_sigmoid,
    "fc_1024_512_bn_dropout": build_fc_1024_512_bn_dropout,
    "fc_tanh": build_fc_tanh,
    "fc_GELU": build_fc_gelu,
    "silu": build_fc_silu,
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


def evaluate(model: nn.Module, loader) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds).astype(np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
    return cm_norm


def main():
    device = torch.device("cuda")

    train_loader = load_dataset("train.csv")
    val_loader = load_dataset("validation.csv")
    test_loader = load_dataset("test.csv")

    loaders = [
        (train_loader, "Treino"),
        (val_loader, "Validação"),
        (test_loader, "Teste"),
    ]

    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    for arch in ARCHITECTURES:
        model_path = os.path.join("results", "ResNet", arch, "model.pth")
        if not os.path.exists(model_path):
            print(f"{arch} weights not found.")
            continue

        model = get_model(arch, device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for col, (loader, title) in enumerate(loaders):
            cm = evaluate(model, loader)
            ax = axes[col]
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f",
                vmin=0.0,
                vmax=1.0,
                cmap="Blues",
                cbar=False,
                xticklabels=CLASSES,
                yticklabels=CLASSES,
                ax=ax,
            )
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            ax.set_title(title)

        fig.suptitle(arch, fontsize=16, y=1.05)
        plt.tight_layout()

        plot_filename = os.path.join(plots_dir, f"confusion_matrix_{arch}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
