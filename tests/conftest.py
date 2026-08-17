"""
Shared fixtures and test doubles for the J.A.R.V.I.S. automated test suite.
"""

import io

import pytest
from PIL import Image


class MockUploadedFile:
    """Mock file upload matching Streamlit UploadedFile interface."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data

    def seek(self, pos: int):
        pass


@pytest.fixture
def sample_text_file():
    return MockUploadedFile("test_doc.txt", "J.A.R.V.I.S. Core Intelligence Engine".encode("utf-8"))


@pytest.fixture
def sample_csv_file():
    csv_content = "Name,Role,Experience\nAlice,Engineer,5\nBob,Data Scientist,3\n"
    return MockUploadedFile("team_data.csv", csv_content.encode("utf-8"))


@pytest.fixture
def sample_image_file():
    img = Image.new("RGB", (320, 240), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return MockUploadedFile("test_frame.jpg", buf.getvalue())


@pytest.fixture
def sample_resume_text():
    return (
        "Senior Software Engineer with 6 years of experience in Python, PyTorch, Docker, "
        "PostgreSQL, AWS, and FastAPI. Master of Science in Computer Science from State University. "
        "Led a team of 8 engineers, architected distributed inference pipeline, reduced latency by 45%, "
        "and scaled system to 2M daily active users."
    )


@pytest.fixture
def sample_job_description():
    return (
        "We are looking for a Senior Software Engineer with 5+ years of experience. "
        "Required: Python, PyTorch, Docker, Kubernetes, PostgreSQL. "
        "Preferred: AWS, CI/CD, Redis. "
        "Education: Master of Science in Computer Science or related field."
    )
