#!/usr/bin/env python3
"""
Main entry point for the Lecture Transcriber package.
Allows running the package with: python -m src
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.cli import main

if __name__ == "__main__":
    main()