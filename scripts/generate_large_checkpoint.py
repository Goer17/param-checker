#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def main() -> None:
    torch.manual_seed(42)
    model = nn.Linear(1024, 1024, bias=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "extra": {
            "big_matrix": torch.randn(512, 2048),
            "preview": torch.randn(16, 16),
        },
    }
    out_path = Path("large.pth")
    torch.save(checkpoint, out_path)
    print(f"Saved large checkpoint to {out_path.resolve()}")


if __name__ == "__main__":
    main()
