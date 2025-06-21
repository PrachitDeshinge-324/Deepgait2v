"""
Warning Suppression Utility for Clean Inference Experience
Suppresses common warnings that don't affect functionality
"""

import warnings
import logging
import os
import sys
from contextlib import contextmanager
from typing import List, Optional

# Configure logging to suppress verbose outputs
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('onnxruntime').setLevel(logging.ERROR)
logging.getLogger('insightface').setLevel(logging.WARNING)

class WarningManager:
    """Manages warning suppression for cleaner output"""
    
    def __init__(self):
        self.suppressed_warnings = []
        self.original_warn = warnings.warn
        
    def suppress_common_warnings(self):
        """Suppress common warnings that don't affect functionality"""
        
        # Suppress torchvision deprecation warnings
        warnings.filterwarnings('ignore', message='.*pretrained.*deprecated.*')
        warnings.filterwarnings('ignore', message='.*Arguments other than a weight enum.*')
        
        # Suppress ONNX runtime warnings
        warnings.filterwarnings('ignore', message='.*Specified provider.*not in available provider names.*')
        warnings.filterwarnings('ignore', category=UserWarning, module='onnxruntime')
        
        # Suppress MPS fallback warnings (these are informational only)
        warnings.filterwarnings('ignore', message='.*operator.*not currently supported on the MPS backend.*')
        
        # Suppress xFormers warnings
        warnings.filterwarnings('ignore', message='.*xFormers not available.*')
        
        # Suppress other common ML warnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
        warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
        
        # Suppress insightface verbose output
        logging.getLogger('insightface').setLevel(logging.CRITICAL)
        
    def restore_warnings(self):
        """Restore warning behavior to default"""
        warnings.resetwarnings()
        warnings.warn = self.original_warn
        
    @contextmanager
    def suppress_stdout_stderr(self):
        """Context manager to suppress stdout and stderr temporarily"""
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
    @contextmanager
    def suppress_warnings_context(self, warning_types: Optional[List[str]] = None):
        """Context manager for temporary warning suppression"""
        if warning_types is None:
            warning_types = ['default']
            
        original_filters = warnings.filters.copy()
        
        try:
            if 'default' in warning_types:
                self.suppress_common_warnings()
            yield
        finally:
            warnings.filters = original_filters

def setup_clean_environment():
    """Setup a clean environment with suppressed warnings"""
    manager = WarningManager()
    manager.suppress_common_warnings()
    
    # Also suppress some specific print statements from libraries
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    return manager

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that filters out known non-critical warnings"""
    
    # Define patterns of warnings to suppress
    suppress_patterns = [
        'pretrained.*deprecated',
        'Arguments other than a weight enum',
        'Specified provider.*not in available provider names',
        'operator.*not currently supported on the MPS backend',
        'xFormers not available'
    ]
    
    message_str = str(message)
    for pattern in suppress_patterns:
        import re
        if re.search(pattern, message_str, re.IGNORECASE):
            return  # Suppress this warning
    
    # For non-suppressed warnings, use default behavior
    warnings._showwarning_orig(message, category, filename, lineno, file, line)

# Set up custom warning handler
if not hasattr(warnings, '_showwarning_orig'):
    warnings._showwarning_orig = warnings.showwarning
    warnings.showwarning = custom_warning_handler
