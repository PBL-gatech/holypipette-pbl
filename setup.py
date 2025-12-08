"""Setup script for the PatcherBot-Agent package."""

from __future__ import annotations

from pathlib import Path
from setuptools import find_packages, setup


def parse_requirements(filename: str) -> list[str]:
    """Load requirements from a pip requirements file."""
    req_path = Path(__file__).parent / filename
    with req_path.open() as req_file:
        return [
            line.strip()
            for line in req_file
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="PatcherBot-Agent",
    version="0.1",
    description="Deep Learning guided Automated Patch Clamp Electrophysiology System",
    url="https://github.com/PBL-gatech/PatcherBot-Agent",
    author="Benjamin Magondu, Nathan Malta, Kaden StillWagon, Victor Guyard",
    author_email="bmagondu3@gatech.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    install_requires=parse_requirements("requirements_runtime.txt"),
)

