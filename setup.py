#!/usr/bin/env python
"""
Setup script for planeLLM package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="planellm",
    version="0.1.0",
    author="OCI DevRel Team",
    author_email="devrel@oracle.com",
    description="Bite-sized podcasts to learn about anything powered by the OCI GenAI Service",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oracle-devrel/planellm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "planellm=podcast_controller:main",
        ],
    },
) 