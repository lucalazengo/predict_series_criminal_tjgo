#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Prophet Forecasting Pipeline package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="prophet-forecasting-pipeline",
    version="1.0.0",
    author="ML Engineering Team",
    author_email="ml-team@example.com",
    description="A comprehensive, automated pipeline for time series forecasting using Facebook Prophet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/prophet-forecasting-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prophet-pipeline=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "prophet",
        "forecasting",
        "time-series",
        "machine-learning",
        "data-science",
        "pipeline",
        "automation",
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/prophet-forecasting-pipeline/issues",
        "Source": "https://github.com/example/prophet-forecasting-pipeline",
        "Documentation": "https://prophet-forecasting-pipeline.readthedocs.io/",
    },
)
