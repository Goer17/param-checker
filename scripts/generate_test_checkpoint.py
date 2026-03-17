#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
        self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main() -> None:
    model = TinyNet()
    checkpoint = {
        "state_dict": model.state_dict(),
        "nested": {"layer": {"bias": torch.randn(6, 6)}},
    }
    out_path = Path("sample.pth")
    torch.save(checkpoint, out_path)
    print(f"Saved checkpoint to {out_path.resolve()}")


if __name__ == "__main__":
    main()
