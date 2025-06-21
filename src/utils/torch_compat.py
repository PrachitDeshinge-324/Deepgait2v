"""
PyTorch Compatibility Bypass for macOS
This module provides a workaround for PyTorch compatibility issues
"""

import sys
import os

# Mock torch module to prevent import errors
class MockTorch:
    def __init__(self):
        self.device = self._device
        self.tensor = self._tensor
        self.zeros = self._zeros
        self.from_numpy = self._from_numpy
        self.load = self._load
        self.backends = MockBackends()
        self.__version__ = "2.1.0"
    
    def _device(self, device_str):
        class MockDevice:
            def __init__(self, device_str):
                self.type = device_str
            def __str__(self):
                return self.type
        return MockDevice(device_str)
    
    def _tensor(self, *args, **kwargs):
        import numpy as np
        if args:
            return np.array(args[0])
        return np.array([])
    
    def _zeros(self, *args, **kwargs):
        import numpy as np
        return np.zeros(args)
    
    def _from_numpy(self, arr):
        return arr
    
    def _load(self, path, **kwargs):
        # Return empty dict for now
        return {}

class MockBackends:
    def __init__(self):
        self.mps = MockMPS()

class MockMPS:
    def is_available(self):
        return False

class MockNN:
    def __init__(self):
        pass

class MockF:
    def __init__(self):
        pass

# Install mock modules if torch import fails
def install_mock_torch():
    """Install mock torch modules to prevent import errors"""
    if 'torch' not in sys.modules:
        sys.modules['torch'] = MockTorch()
        sys.modules['torch.nn'] = MockNN()
        sys.modules['torch.nn.functional'] = MockF()
        print("🔄 Using PyTorch compatibility mode (geometric parsing only)")

# Check if torch can be imported, if not use mock
try:
    import torch
    print("✅ PyTorch available")
except ImportError:
    install_mock_torch()
    print("⚠️  PyTorch not available, using compatibility mode")
