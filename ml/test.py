# ml/test.py

import numpy as np
import torch
from common import build_transforms


def test_transforms_shape():
    tf = build_transforms(224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), train=False)
    dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
    result = tf(image=dummy_image)["image"]
    assert result.shape == (3, 224, 224), f"Expected (3,224,224), got {result.shape}"
    print("test_transforms_shape passed")


def test_gate_embedding_shape():
    from build_gate import get_embedding
    from transformers import AutoModel, AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    size = processor.crop_size["height"]
    dummy = torch.zeros(1, 3, size, size)
    emb = get_embedding(model, dummy)
    assert emb.shape[-1] == 768, f"Expected 768-dim embedding, got {emb.shape}"
    print("test_gate_embedding_shape passed")


if __name__ == "__main__":
    test_transforms_shape()
    test_gate_embedding_shape()
    print("All tests passed.")
