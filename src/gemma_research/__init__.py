"""Gemma DeepResearch E4B runtime."""

from .agent import ResearchAgent
from .config import Config, load_config

__all__ = ["Config", "ResearchAgent", "load_config"]

__version__ = "0.1.0"
