"""
Source package for the gait recognition system
"""

# Core components
from . import core
from . import models
from . import processing
from . import identification
from . import adapters
from . import app
from . import utils

__all__ = [
    'core',
    'models', 
    'processing',
    'identification',
    'adapters',
    'app',
    'utils'
]
