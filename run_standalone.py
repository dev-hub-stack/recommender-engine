#!/usr/bin/env python3
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

# Import and run
from src.main import app
from shared.config.settings import logging_settings
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level=logging_settings.level.lower()
    )
