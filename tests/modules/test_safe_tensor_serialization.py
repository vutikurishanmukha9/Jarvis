"""
Tests for Safe PyTorch Tensor Embedding Serialization (Eliminating Python Pickle).
"""

import os
import torch
import pytest
from src.modules.career.scorer.config import EMBEDDING_CACHE_FILE

def test_safe_tensor_cache_file_extension():
    """Verify that EMBEDDING_CACHE_FILE uses safe .pt PyTorch format."""
    assert str(EMBEDDING_CACHE_FILE).endswith(".pt")

def test_safe_torch_save_and_load(tmp_path):
    """Verify saving and loading embeddings with weights_only=True."""
    dummy_embeddings = torch.randn(10, 384)
    save_path = tmp_path / "test_embeddings.pt"

    torch.save(dummy_embeddings, save_path)
    assert save_path.exists()

    loaded = torch.load(save_path, map_location="cpu", weights_only=True)
    assert loaded.shape == (10, 384)
    assert torch.allclose(dummy_embeddings, loaded)
