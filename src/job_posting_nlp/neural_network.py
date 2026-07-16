from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HIDDEN_DIMENSION = 128
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class NeuralFoldResult:
    predictions: NDArray[np.int64]
    final_training_loss: float
    parameter_count: int


class SmallEmbeddingClassifier(nn.Module):
    """Small feed-forward classifier for fixed dense embeddings."""

    def __init__(
        self,
        *,
        input_dimension: int,
        output_dimension: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMENSION, output_dimension),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def calculate_balanced_class_weights(
    labels: NDArray[np.int64],
    *,
    class_count: int,
) -> NDArray[np.float32]:
    """Calculate the balanced weights used by cross-entropy loss."""
    label_counts = np.bincount(labels, minlength=class_count)
    weights = len(labels) / (class_count * label_counts)

    return weights.astype(np.float32)


def fit_small_embedding_classifier(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    train_indices: NDArray[np.int64],
    validation_indices: NDArray[np.int64],
    *,
    class_count: int,
    device: str,
    seed: int,
) -> NeuralFoldResult:
    """Fit the fixed neural classifier for one cross-validation fold."""
    torch.manual_seed(seed)

    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model = SmallEmbeddingClassifier(
        input_dimension=embeddings.shape[1],
        output_dimension=class_count,
    ).to(device)

    training_labels = labels[train_indices]
    class_weights = calculate_balanced_class_weights(
        training_labels,
        class_count=class_count,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    training_dataset = TensorDataset(
        torch.from_numpy(embeddings[train_indices]).float(),
        torch.from_numpy(training_labels).long(),
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    final_training_loss = np.nan

    for _ in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for features, targets in training_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = loss_function(
                model(features),
                targets,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(features)

        final_training_loss = (
            total_loss / len(training_dataset)
        )

    model.eval()

    validation_features = torch.from_numpy(
        embeddings[validation_indices]
    ).float().to(device)

    with torch.inference_mode():
        predictions = (
            model(validation_features)
            .argmax(dim=1)
            .cpu()
            .numpy()
            .astype(np.int64)
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    return NeuralFoldResult(
        predictions=predictions,
        final_training_loss=float(
            final_training_loss
        ),
        parameter_count=parameter_count,
    )