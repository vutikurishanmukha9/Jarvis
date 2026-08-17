"""
Specialized multimodal intelligence engines for Jarvis:
- vision: YOLOv8 object localization, OpenCV image metrics & OCR
- career: ATS compatibility scoring, skills extraction & salary projection
- outreach: Smart HR recruiter outreach, tag personalization & batch dispatch
"""

from .career import get_career_tools
from .outreach import get_outreach_tools
from .vision import get_vision_tools

__all__ = ["get_vision_tools", "get_career_tools", "get_outreach_tools"]
