#!/usr/bin/env python
"""Setup script for the planeLLM package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="planellm",
    version="0.1.0",
    author="OCI DevRel Team",
    author_email="devrel@oracle.com",
    description="Bite-sized podcasts to learn about anything powered by the OCI GenAI Service",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jasperan/planeLLM",
    license="UPL-1.0",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "local-tts": [
            "torch==2.10.0",
            "transformers>=4.46,<5.0",
            "TTS==0.22.0",
            "flash-attn==2.8.3",
            "parler-tts @ git+https://github.com/huggingface/parler-tts.git@d108732cd57788ec86bc857d99a6cabd66663d68",
        ]
    },
    entry_points={
        "console_scripts": [
            "planellm=podcast_controller:main",
        ],
    },
)
