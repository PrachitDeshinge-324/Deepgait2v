"""
Main entry point for person detection, tracking, and gait recognition with enhanced CCTV accuracy
"""

# Initialize clean environment with warning suppression
from src.utils.warning_suppressor import setup_clean_environment
warning_manager = setup_clean_environment()

import warnings
import numpy as np
import os
import config
from src.app.gait_recognition_app import GaitRecognitionApp


# Enable batch processing optimization
os.environ['ENABLE_BATCH_PROCESSING'] = '1'
os.environ['ENABLE_DEVICE_OPTIMIZATION'] = '1'
    
def main():
    """Main function to run the tracking and recognition application with quality control"""
    app = GaitRecognitionApp(config)
    app.initialize()
    app.run()
    app.cleanup()
    app.report_statistics()

if __name__ == "__main__":
    main()