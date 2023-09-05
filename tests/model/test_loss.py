import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from smart_tree.model.loss import FocalLoss
import pytest


@pytest.fixture
def focal_loss_instance():
    return FocalLoss(gamma=2.0, reduction="mean")


def test_focal_loss(focal_loss_instance):
    # Simulated input logits (log-softmax) and target labels
    input_logits = torch.tensor(
        [[0.5, 1.0, -0.2], [2.0, -0.5, 1.5]], requires_grad=True
    )
    target_labels = torch.tensor([1, 2])  # Corresponding target labels for each sample

    # Calculate the loss
    loss = focal_loss_instance(input_logits, target_labels)

    # Define the expected loss (manually calculated)
    expected_loss = (
        -1
        * (
            (1 - torch.exp(input_logits[0, 1])) ** 2
            * torch.log(torch.exp(input_logits[0, 1]))
        )
        - 1
        * (
            (1 - torch.exp(input_logits[1, 2])) ** 2
            * torch.log(torch.exp(input_logits[1, 2]))
        )
        / 2
    )

    # Check if the calculated loss matches the expected loss
    assert torch.allclose(loss, expected_loss, atol=1e-5)


if __name__ == "__main__":
    pytest.main()
