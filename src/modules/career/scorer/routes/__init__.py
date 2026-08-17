"""
Routes package for AI Resume Analyzer

Imports and re-exports all route routers.
"""

from src.modules.career.scorer.routes.analyze import router as analyze_router
from src.modules.career.scorer.routes.ats import router as ats_router
from src.modules.career.scorer.routes.general import router as general_router
from src.modules.career.scorer.routes.match import router as match_router
from src.modules.career.scorer.routes.upload import router as upload_router

__all__ = ["general_router", "upload_router", "match_router", "ats_router", "analyze_router"]
