# -*- coding: utf-8 -*-
"""Test configuration - ensure project root is on sys.path."""
import sys
import os

# Add project root to path so torch_core can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
