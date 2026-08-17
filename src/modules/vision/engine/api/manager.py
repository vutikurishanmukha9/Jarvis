import asyncio
import uuid
import logging
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any
from .schemas import JobStatus, JobResponse

from ..multimodal_system import MultimodalAI

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
        self._prune_jobs()
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

    def _prune_jobs(self, max_jobs: int = 1_000) -> None:
        """Bound in-memory job history, retaining pending work and recent completions."""
        if len(self.jobs) < max_jobs:
            return
        completed = sorted(
            (job for job in self.jobs.values() if job["completed_at"] is not None),
            key=lambda job: job["completed_at"],
        )
        for job in completed[: max(0, len(self.jobs) - max_jobs + 1)]:
            self.jobs.pop(job["id"], None)

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
        """Process queued jobs, grouping only requests with equivalent configuration."""
        groups: Dict[str, list] = {}
        for item in batch:
            key = json.dumps(item["config"], sort_keys=True, default=str)
            groups.setdefault(key, []).append(item)

        for group in groups.values():
            job_ids = [item["job_id"] for item in group]
            image_paths = [item["image_path"] for item in group]
            questions = [item["question"] for item in group]
            config = group[0]["config"]
            logger.info("Processing %d job(s): %s", len(group), job_ids)
            for job_id in job_ids:
                self.jobs[job_id]["status"] = JobStatus.PROCESSING

            try:
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(
                    self.executor, self.system.process_batch, image_paths, questions, config
                )
                if len(results) != len(group):
                    raise RuntimeError("Vision backend returned an unexpected result count.")
                for item, result in zip(group, results):
                    job = self.jobs[item["job_id"]]
                    if result.get("error"):
                        job["status"] = JobStatus.FAILED
                        job["error"] = result["error"]
                    else:
                        job["status"] = JobStatus.COMPLETED
                        job["result"] = result
                    job["completed_at"] = datetime.now()
            except Exception as exc:
                logger.exception("Vision batch processing failed")
                for job_id in job_ids:
                    job = self.jobs[job_id]
                    job["status"] = JobStatus.FAILED
                    job["error"] = str(exc)
                    job["completed_at"] = datetime.now()
            finally:
                for image_path in image_paths:
                    try:
                        os.remove(image_path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        logger.warning("Could not remove temporary upload: %s", image_path)
