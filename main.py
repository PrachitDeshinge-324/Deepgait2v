"""
Main entry point for person detection, tracking, and gait recognition with enhanced CCTV accuracy
"""

import config
from app.gait_recognition_app import GaitRecognitionApp

def main():
    """Main function to run the tracking and recognition application with quality control"""
    app = GaitRecognitionApp(config)
    app.initialize()
    app.run()
    app.cleanup()
    app.report_statistics()

if __name__ == "__main__":
    main()