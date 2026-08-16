import logging
import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json

from .schemas import AnalysisRequest, JobResponse, AnalysisConfig
from .manager import JobManager
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_system import MultimodalAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
job_manager: Optional[JobManager] = None

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    config: str = Form(None), # JSON string for config
    manager: JobManager = Depends(get_job_manager)
):
    """
    Upload an image and start analysis.
    Returns a job ID to poll for results.
    """
    # Create temp file for the image
    try:
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        file_location = os.path.join(temp_dir, file.filename)
        
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # Parse config
        analysis_config = None
        if config:
            try:
                analysis_config = json.loads(config)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in config")

        # Submit job
        job_id = await manager.submit_job(file_location, question, analysis_config)
        
        # Get initial status
        return manager.get_job(job_id)
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    manager: JobManager = Depends(get_job_manager)
):
    """Get the status and result of a job."""
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/health")
async def health_check():
    return {"status": "ok"}
