"""
Model Manager - Centralized ML model loading and management
"""
import os
import hashlib
import logging
import traceback
import threading
from typing import Optional, List, Tuple, Any, Dict

import torch
import joblib
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.modules.career.scorer.config import (
    JOB_CLASSIFIER_PATH,
    SALARY_PREDICTOR_PATH,
    JOB_DATA_CSV,
    EMBEDDING_CACHE_FILE
)

logger = logging.getLogger(__name__)

# Immutable release-artifact checksums.  Update these only as part of the
# reviewed model-release process; never load an artifact that fails validation.
MODEL_ARTIFACT_SHA256 = {
    "job_classifier.pkl": "B46E1CF27827CB801A7A4D0E9ED873B0B23B5D51FD8964A27620E9F237EA71BF",
    "salary_predictor.pkl": "28F054806C0BD3746EEA21156A05423E743B17B664C6603F6585C4CEA2A50393",
}


def _verify_model_artifact(path: str) -> None:
    """Require exact release checksums before deserializing legacy model files."""
    expected = MODEL_ARTIFACT_SHA256.get(os.path.basename(path))
    if expected is None:
        raise ValueError(f"No approved checksum is configured for model artifact: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().upper() != expected:
        raise ValueError(f"Model artifact checksum validation failed: {path}")


class ModelManager:
    """Centralized model management with FAISS ANN indexing and safe tensor serialization"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.resume_classifier = None
        self.salary_model = None
        self.job_df = None
        self.embed_model = None
        self.job_embeddings = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self._models_loaded = False
        self._initialized = True

    def load_models(self):
        """Load all ML models with comprehensive error handling"""
        if self._models_loaded:
            logger.info("Models already loaded, skipping...")
            return

        try:
            logger.info("Loading models...")

            # Load classifier
            if not os.path.exists(JOB_CLASSIFIER_PATH):
                raise FileNotFoundError(f"job_classifier.pkl not found at {JOB_CLASSIFIER_PATH}")
            _verify_model_artifact(JOB_CLASSIFIER_PATH)
            self.resume_classifier = joblib.load(JOB_CLASSIFIER_PATH)
            logger.info("Resume classifier loaded")

            # Load salary predictor
            if not os.path.exists(SALARY_PREDICTOR_PATH):
                raise FileNotFoundError(f"salary_predictor.pkl not found at {SALARY_PREDICTOR_PATH}")
            _verify_model_artifact(SALARY_PREDICTOR_PATH)
            self.salary_model = joblib.load(SALARY_PREDICTOR_PATH)
            logger.info("Salary predictor loaded")

            # Load job dataset
            if not os.path.exists(JOB_DATA_CSV):
                raise FileNotFoundError(f"job_title_des.csv not found at {JOB_DATA_CSV}")
            self.job_df = pd.read_csv(JOB_DATA_CSV)

            # Validate dataset columns
            required_columns = ['Job Description', 'Job Title']
            if not all(col in self.job_df.columns for col in required_columns):
                raise ValueError(f"Dataset must contain {required_columns} columns")

            # Remove any rows with missing critical data
            self.job_df = self.job_df.dropna(subset=required_columns)
            logger.info(f"Job dataset loaded with {len(self.job_df)} entries")

            # Load embedding model
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embed_model.max_seq_length = 256
            logger.info("Sentence Transformer model loaded")

            # Precompute embeddings
            self._precompute_job_embeddings()

            self._models_loaded = True
            logger.info("All models successfully initialized")

        except FileNotFoundError as e:
            logger.error(f"Required file missing: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def _precompute_job_embeddings(self):
        """Precompute embeddings for job descriptions with validation"""
        try:
            # Try to load cached embeddings safely using PyTorch weights_only=True
            if os.path.exists(EMBEDDING_CACHE_FILE):
                try:
                    self.job_embeddings = torch.load(
                        EMBEDDING_CACHE_FILE,
                        map_location="cpu",
                        weights_only=True
                    )

                    # Validate cache matches current dataset
                    if len(self.job_embeddings) == len(self.job_df):
                        logger.info("Loaded cached job embeddings via safe torch.load")
                        self._build_faiss_index()
                        return
                    else:
                        logger.warning("Cache size mismatch, recomputing embeddings...")
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}, recomputing embeddings...")

            # Compute new embeddings
            job_descriptions = self.job_df['Job Description'].fillna('').tolist()

            if not job_descriptions:
                raise ValueError("No job descriptions found in dataset")

            logger.info(f"Computing embeddings for {len(job_descriptions)} job descriptions...")
            self.job_embeddings = self.embed_model.encode(
                job_descriptions,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32
            )

            # Cache the embeddings safely using PyTorch
            torch.save(self.job_embeddings, EMBEDDING_CACHE_FILE)
            logger.info("Job embeddings computed and cached via torch.save")

            # Build in-memory FAISS ANN index
            self._build_faiss_index()

        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            raise

    def _build_faiss_index(self):
        """Build and populate a FAISS IndexFlatIP index with L2-normalized vectors."""
        try:
            if self.job_embeddings is None:
                return

            if isinstance(self.job_embeddings, torch.Tensor):
                embeddings_np = self.job_embeddings.detach().cpu().numpy().astype("float32")
            else:
                embeddings_np = np.array(self.job_embeddings, dtype="float32")

            # L2 normalize for cosine similarity via inner product
            faiss.normalize_L2(embeddings_np)
            dimension = embeddings_np.shape[1]

            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_np)
            self.faiss_index = index
            logger.info(f"FAISS IndexFlatIP initialized with {index.ntotal} vectors (dim={dimension})")
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")

    def search_jobs(self, query_embedding: Any, top_k: int = 3) -> List[Tuple[str, float, int]]:
        """
        Query the FAISS vector index for top-k matching jobs in sub-millisecond C++ time.
        
        Returns:
            List of (job_title, similarity_score, dataset_row_index)
        """
        if self.faiss_index is None:
            self._build_faiss_index()

        if self.faiss_index is None:
            return []

        if isinstance(query_embedding, torch.Tensor):
            query_np = query_embedding.detach().cpu().numpy().astype("float32")
        else:
            query_np = np.array(query_embedding, dtype="float32")

        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        faiss.normalize_L2(query_np)
        distances, indices = self.faiss_index.search(query_np, top_k)

        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.job_df):
                continue
            job_title = self.job_df.iloc[idx]["Job Title"]
            matches.append((job_title, float(dist), int(idx)))

        return matches

    def is_loaded(self):
        """Check if models are loaded"""
        return self._models_loaded


# Singleton instance
model_manager = ModelManager()


def load_all_models():
    """Load all ML models synchronously during startup.
    
    Raises on failure so FastAPI's lifespan handler aborts
    instead of starting a silently broken server.
    """
    try:
        logger.info("Starting model loading...")
        model_manager.load_models()
        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(traceback.format_exc())
        raise  # Let lifespan handler abort startup


# Backward-compat alias
load_models_background = load_all_models


