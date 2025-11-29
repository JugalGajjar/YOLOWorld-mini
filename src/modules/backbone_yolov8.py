from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOv8Backbone(nn.Module):
    """
    Wraps a YOLOv8 model and exposes the multi-scale feature maps that feed into the 
    Detect head (typically three scales).
    """

    def __init__(self, variant: str = "yolov8s.pt", pretrained: bool = True):
        super().__init__()

        # Load YOLOv8 model
        yolo = YOLO(variant)
        task_model = yolo.model  # BaseModel / DetectionModel

        if not hasattr(task_model, "model"):
            raise RuntimeError("Unexpected YOLO model structure: missing 'model' attribute.")

        layers = task_model.model  # nn.ModuleList
        if len(layers) == 0:
            raise RuntimeError("YOLO model has no layers in .model.")

        detect_layer = layers[-1]

        if not isinstance(detect_layer, nn.Module):
            raise RuntimeError("Last layer of YOLO model is not an nn.Module.")

        self.task_model = task_model
        self.detect_layer = detect_layer

        # This will store the features captured by the hook
        self._captured_feats: Optional[List[torch.Tensor]] = None

        # Register forward hook on the Detect head
        self.detect_layer.register_forward_hook(self._detect_forward_hook)

    def _detect_forward_hook(self, module: nn.Module, inputs, output):
        """
        Forward hook on the Detect head. Captures the input feature maps
        that are fed into the Detect layer during a forward pass.
        """
        if not inputs:
            self._captured_feats = None
            return

        x = inputs[0]  # this should be a list of tensors
        if isinstance(x, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in x):
            self._captured_feats = list(x)
        else:
            # Unexpected format; keep for debugging
            self._captured_feats = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a normal YOLOv8 forward pass, but return only the feature maps (P3, P4, P5), 
        each a tensor (B, C_i, H_i, W_i), that are fed into the Detect head.
        """
        # Reset captured features
        self._captured_feats = None

        # Run full model forward to trigger the Detect head and its hook
        _ = self.task_model(x)

        if self._captured_feats is None:
            raise RuntimeError(
                "Detect head hook did not capture any features. "
                "The YOLO model structure may be different than expected."
            )

        feats = self._captured_feats

        if len(feats) < 3:
            raise RuntimeError(
                f"Expected at least 3 feature maps from Detect head input, "
                f"but got {len(feats)}."
            )

        # We take the last 3 detection scales
        p3, p4, p5 = feats[-3], feats[-2], feats[-1]
        return p3, p4, p5


# Simple test
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    model = YOLOv8Backbone(variant="models/yolov8s.pt")
    model.to(device)
    model.eval()

    x = torch.randn(1, 3, 640, 640, device=device)

    with torch.no_grad():
        p3, p4, p5 = model(x)

    print("P3 shape:", p3.shape)
    print("P4 shape:", p4.shape)
    print("P5 shape:", p5.shape)