from typing import Tuple

import torch


class APLoss(torch.nn.Module):
    """! Quantized average precision loss.

    Implementation based on 'Learning with Average Precision: Training Image Retrieval with a Listwise Loss'
    by Revaud et al.
    """

    def __init__(self, num_bins: int = 20, value_range: Tuple[float, float] = (-1.0, 1.0)) -> None:
        """! Class initializer.

        @param num_bins Number of quantization bins (= M).
        @param value_range The range of the values to quantize.
        """
        super().__init__()
        min_value, max_value = value_range
        self.delta = (max_value - min_value) / (num_bins - 1)  # bin interval
        self.bin_centers = max_value - self.delta * torch.arange(0, num_bins)

    def forward(self, sim: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """! Forward pass.

        @param sim Similarity values (B, N).
        @param y Ground truth labels (B, N).
        @return The loss 1 - mAP.
        """
        # Triangular kernel (eq 7)
        delta = torch.clamp(
            1.0 - torch.abs(sim.unsqueeze(2) - self.bin_centers.to(sim.device)) / self.delta, min=0.0
        )  # (B, N, M)

        # Precision and incremental recall (eq 9 & 10)
        recall = (delta * y.unsqueeze(2)).sum(dim=1)  # (B, M)
        precision = recall.cumsum(dim=1) / (1e-16 + delta.sum(dim=1).cumsum(dim=1))  # (B, M)
        recall /= y.sum(dim=1).unsqueeze(1)  # (B, M)

        # AP (eq 11)
        ap = (precision * recall).sum(dim=1)  # B

        return 1.0 - ap.mean()
