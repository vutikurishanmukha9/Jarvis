import asyncio
import uuid
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any
from .schemas import JobStatus, JobResponse
import sys
import os

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_system import MultimodalAI

logger = logging.getLogger(__name__)

class JobManager:
    """
    Manages asynchronous analysis jobs.
    Uses a thread pool to run the CPU/GPU intensive tasks without blocking the async event loop.
    """
    def __init__(self, model_system: MultimodalAI, batch_size: int = 4, max_latency: float = 0.5):
        self.system = model_system
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=1) # Single worker for batch processing (thread safe)
        
        # Batching configuration
        self.batch_size = batch_size
        self.max_latency = max_latency
        self.queue = asyncio.Queue()
        self.processing_task = None
        self.running = False

    async def start(self):
        """Start the batch processing loop."""
        self.running = True
        self.processing_task = asyncio.create_task(self._batch_loop())
        logger.info("Batch processing loop started")

    async def stop(self):
        """Stop the batch processing loop."""
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown()
        logger.info("Batch processing loop stopped")

    def create_job(self) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "created_at": datetime.now(),
            "completed_at": None,
            "error": None,
            "result": None
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        """Get the status of a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        return JobResponse(
            job_id=job["id"],
            status=job["status"],
            created_at=job["created_at"],
            completed_at=job["completed_at"],
            error=job["error"],
            result=job["result"]
        )

    async def submit_job(self, image_path: str, question: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Submit a job for processing via the batch queue."""
        job_id = self.create_job()
        
        # Add to queue
        await self.queue.put({
            'job_id': job_id,
            'image_path': image_path,
            'question': question,
            'config': config
        })
        
        return job_id

    async def _batch_loop(self):
        """Infinite loop to process batches."""
        logger.info("Batch loop active")
        while self.running:
            batch = []
            try:
                # 1. Wait for first item (blocking)
                item = await self.queue.get()
                batch.append(item)
                
                # 2. Try to fill batch within max_latency
                deadline = asyncio.get_event_loop().time() + self.max_latency
                
                while len(batch) < self.batch_size:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break
                        
                    try:
                        # Non-blocking peek/get with timeout
                        item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                # 3. Process the batch
                if batch:
                    await self._process_batch(batch)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch loop: {e}")
                await asyncio.sleep(1) # Prevent tight loop on error

    async def _process_batch(self, batch: list):
        """Process a list of job items."""
        job_ids = [item['job_id'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        questions = [item['question'] for item in batch]
        # Use config from first item for the whole batch for now
        # Ideally, we should group by config, but for simplicity assuming uniform config
        config = batch[0]['config'] 

        logger.info(f"Processing batch of {len(batch)} jobs: {job_ids}")
        
        # Mark all as processing
        for jid in job_ids:
            self.jobs[jid]['status'] = JobStatus.PROCESSING

        try:
            # Run blocking inference in executor
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                self.executor,
                self.system.process_batch,
                image_paths,
                questions,
                config
            )
            
            # Map results back to jobs
            for i, jid in enumerate(job_ids):
                if results[i].get('error'):
                    self.jobs[jid]['status'] = JobStatus.FAILED
                    self.jobs[jid]['error'] = results[i]['error']
                else:
                    self.jobs[jid]['status'] = JobStatus.COMPLETED
                    self.jobs[jid]['result'] = results[i]
                self.jobs[jid]['completed_at'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for jid in job_ids:
                self.jobs[jid]['status'] = JobStatus.FAILED
                self.jobs[jid]['error'] = str(e)
                self.jobs[jid]['completed_at'] = datetime.now() 

    def create_job(self) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "created_at": datetime.now(),
            "completed_at": None,
            "error": None,
            "result": None
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        """Get the status of a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        return JobResponse(
            job_id=job["id"],
            status=job["status"],
            created_at=job["created_at"],
            completed_at=job["completed_at"],
            error=job["error"],
            result=job["result"]
        )

    async def submit_job(self, image_path: str, question: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Submit a job for processing."""
        job_id = self.create_job()
        
        # Run the processing in a separate thread
        asyncio.create_task(self._process_job(job_id, image_path, question, config))
        
        return job_id

    async def _process_job(self, job_id: str, image_path: str, question: str, config: Optional[Dict[str, Any]] = None):
        """Internal method to process the job in background."""
        logger.info(f"Starting job {job_id} for image {image_path}")
        
        self.jobs[job_id]["status"] = JobStatus.PROCESSING
        
        try:
            # Run the synchronous process method in the thread pool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.system.process,
                image_path,
                question,
                config
            )
            
            if result.get('error'):
                self.jobs[job_id]["status"] = JobStatus.FAILED
                self.jobs[job_id]["error"] = result['error']
            else:
                self.jobs[job_id]["status"] = JobStatus.COMPLETED
                self.jobs[job_id]["result"] = result
                
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            self.jobs[job_id]["status"] = JobStatus.FAILED
            self.jobs[job_id]["error"] = str(e)
            
        finally:
            self.jobs[job_id]["completed_at"] = datetime.now()
