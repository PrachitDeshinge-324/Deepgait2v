"""
Main entry point for person detection, tracking, and gait recognition with enhanced CCTV accuracy
"""

import warnings
import numpy as np

# Suppress only the specific InsightFace rcond warning to avoid interfering with tqdm
warnings.filterwarnings("ignore", 
                       message=".*rcond.*parameter will change.*", 
                       category=FutureWarning,
                       module=".*insightface.*")

import config
from src.app.gait_recognition_app import GaitRecognitionApp

def main():
    """Main function to run the tracking and recognition application with quality control"""
    app = GaitRecognitionApp(config)
    app.initialize()
    app.run()
    app.cleanup()
    app.report_statistics()

if __name__ == "__main__":
    main()