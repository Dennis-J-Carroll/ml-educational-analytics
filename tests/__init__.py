"""
Test suite for Educational Analytics Dashboard

This package contains comprehensive unit tests for all components
of the educational analytics system including:
- Configuration management tests
- Database operation tests  
- Analytics engine tests
- Visualization component tests
- Utility function tests
"""

import pytest
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))