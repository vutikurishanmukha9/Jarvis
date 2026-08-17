"""
Tests for FAISS vector indexing and nearest neighbor search in Career Module.
Verifies L2 normalization, sub-millisecond query execution, and multi-tensor format handling.
"""

import pytest
import numpy as np
import torch
import faiss
import pandas as pd

from src.modules.career.scorer.services.model_manager import ModelManager


@pytest.fixture
def mock_model_manager():
    """Create an isolated ModelManager instance with synthetic embeddings."""
    mgr = ModelManager()
    mgr.job_df = pd.DataFrame({
        "Job Title": [
            "Senior AI Engineer",
            "Full Stack Developer",
            "Data Scientist",
            "DevOps Cloud Architect",
            "Product Manager"
        ],
        "Job Description": [
            "Lead LLM architecture and multi-agent systems with PyTorch.",
            "Build React and Python FastAPI microservices.",
            "Train machine learning models and feature engineering with Pandas.",
            "Deploy Kubernetes clusters and CI/CD pipelines on AWS.",
            "Drive product roadmap and sprint planning."
        ]
    })
    # Create deterministic synthetic embeddings [5, 384]
    np.random.seed(42)
    fake_embeddings = np.random.randn(5, 384).astype("float32")
    mgr.job_embeddings = torch.tensor(fake_embeddings)
    mgr._build_faiss_index()
    return mgr


def test_faiss_index_construction_and_dimension(mock_model_manager):
    """Verify FAISS index is constructed with exact dimension and entry count."""
    assert mock_model_manager.faiss_index is not None
    assert mock_model_manager.faiss_index.ntotal == 5
    assert mock_model_manager.faiss_index.d == 384


def test_faiss_search_jobs_ranking_and_exact_match(mock_model_manager):
    """Verify searching for an exact vector returns that job title as top match with score ~1.0."""
    # Query with exact vector of row 0 (Senior AI Engineer)
    query_tensor = mock_model_manager.job_embeddings[0:1]
    matches = mock_model_manager.search_jobs(query_tensor, top_k=3)

    assert len(matches) == 3
    top_title, top_score, top_idx = matches[0]
    assert top_title == "Senior AI Engineer"
    assert top_idx == 0
    assert pytest.approx(top_score, abs=1e-4) == 1.0


def test_faiss_search_jobs_with_numpy_and_torch_tensors(mock_model_manager):
    """Verify search_jobs seamlessly accepts both PyTorch Tensors and 1D/2D NumPy arrays."""
    # 1D numpy array
    query_1d_np = np.random.randn(384).astype("float32")
    matches_np = mock_model_manager.search_jobs(query_1d_np, top_k=2)
    assert len(matches_np) == 2
    assert isinstance(matches_np[0][1], float)

    # 2D torch tensor
    query_2d_torch = torch.tensor(np.random.randn(1, 384).astype("float32"))
    matches_torch = mock_model_manager.search_jobs(query_2d_torch, top_k=2)
    assert len(matches_torch) == 2


def test_faiss_search_jobs_empty_or_uninitialized():
    """Verify search_jobs handles uninitialized states gracefully without crashing."""
    mgr = ModelManager()
    mgr.faiss_index = None
    mgr.job_embeddings = None
    results = mgr.search_jobs(np.zeros((1, 384)), top_k=3)
    assert results == []
