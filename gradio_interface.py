#!/usr/bin/env python
"""
Launcher script for the planeLLM Gradio interface.

This script provides a simple way to launch the Gradio interface
without having to import the module directly.

Usage:
    python gradio_interface.py
"""

# Import directly from gradio_app.py
from gradio_app import create_interface

if __name__ == "__main__":
    print("Starting planeLLM Gradio interface...")
    create_interface() 