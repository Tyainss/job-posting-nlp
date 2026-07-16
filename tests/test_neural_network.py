import numpy as np
import torch

from job_posting_nlp.neural_network import (
    SmallEmbeddingClassifier,
    calculate_balanced_class_weights,
)


def test_small_embedding_classifier_output_shape():
    classifier = SmallEmbeddingClassifier(
        input_dimension=4,
        output_dimension=3,
    )

    logits = classifier(torch.zeros((2, 4)))

    assert logits.shape == (2, 3)


def test_balanced_class_weights_use_training_label_counts():
    labels = np.array(
        [0, 0, 0, 1],
        dtype=np.int64,
    )

    weights = calculate_balanced_class_weights(
        labels,
        class_count=2,
    )

    np.testing.assert_allclose(
        weights,
        np.array(
            [2 / 3, 2],
            dtype=np.float32,
        ),
    )