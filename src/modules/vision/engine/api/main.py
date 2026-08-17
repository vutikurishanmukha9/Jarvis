import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ..multimodal_system import MultimodalAI
from .manager import JobManager
from .schemas import JobResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
job_manager: Optional[JobManager] = None
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global job_manager
    logger.info("Initializing Multimodal AI System...")
    # Initialize the heavy model system once on startup
    # In a real production system, this might be a separate service or loaded lazily
    system = MultimodalAI(device="auto")
    job_manager = JobManager(system)
    await job_manager.start()
    logger.info("System initialized and ready.")
    yield
    # Shutdown
    logger.info("Shutting down...")
    if job_manager:
        await job_manager.stop()
        job_manager.system.cleanup()


app = FastAPI(title="Multimodal AI API", lifespan=lifespan)

# CORS configuration
allowed_origins = [
    origin.strip()
    for origin in os.getenv("JARVIS_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_job_manager():
    if not job_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    return job_manager


@app.post("/analyze", response_model=JobResponse)
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form("Describe this image in detail"),
    config: str = Form(None),  # JSON string for config
    manager: JobManager = Depends(get_job_manager),
):
    """
    Upload an image and start analysis.
    Returns a job ID to poll for results.
    """
    file_location: Optional[Path] = None
    submitted = False
    try:
        temp_dir = Path.cwd() / "temp_uploads"
        temp_dir.mkdir(mode=0o700, exist_ok=True)
        suffix = Path(file.filename or "").suffix.lower()
        file_location = temp_dir / f"{uuid.uuid4().hex}{suffix}"

        total_bytes = 0
        with file_location.open("wb") as file_object:
            while chunk := await file.read(64 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded image exceeds the 20MB limit.")
                file_object.write(chunk)

        # Parse config
        analysis_config = None
        if config:
            try:
                analysis_config = json.loads(config)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON in config") from exc

        # Submit job
        job_id = await manager.submit_job(str(file_location), question, analysis_config)
        submitted = True

        # Get initial status
        return manager.get_job(job_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if file_location is not None and not submitted:
            try:
                file_location.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove failed upload: %s", file_location)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, manager: JobManager = Depends(get_job_manager)):
    """Get the status and result of a job."""
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/health")
async def health_check():
    return {"status": "ok"}
