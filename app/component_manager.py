"""
Component management for the gait recognition system
"""

import logging
from modules.detector import PersonDetector
from modules.tracker import PersonTracker
from modules.visualizer import Visualizer
from modules.silhouette_extractor import SilhouetteExtractor
from modules.gait_recognizer import GaitRecognizer
from modules.quality_assessor import GaitSequenceQualityAssessor
from modules.enhanced_identifier import CCTVGaitIdentifier
from modules.multimodal_identifier import MultiModalIdentifier
from modules.face_embedding_extractor import FaceEmbeddingExtractor
from utils.device import vprint

class ComponentManager:
    """Manages initialization and access to system components"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.detector = None
        self.tracker = None
        self.visualizer = None
        self.silhouette_extractor = None
        self.gait_recognizer = None
        self.quality_assessor = None
        self.face_extractor = None
        self.enhanced_identifier = None
        
    def initialize(self):
        """Initialize all components"""
        # Initialize core components
        self.detector = PersonDetector(self.config.MODEL_PATH)
        self.tracker = PersonTracker(self.config.TRACKER_CONFIG)
        self.visualizer = Visualizer(self.config.VISUALIZATION_CONFIG)
        self.silhouette_extractor = SilhouetteExtractor(
            self.config.SILHOUETTE_CONFIG, 
            self.config.SEG_MODEL_PATH
        )
        
        # Initialize quality assessor
        self.quality_assessor = GaitSequenceQualityAssessor(self.config.QUALITY_ASSESSOR_CONFIG)
        
        # Initialize gait recognizer
        selected_model = getattr(self.config, 'GAIT_MODEL_TYPE', 'DeepGaitV2')
        vprint(f"Initializing GaitRecognizer with {selected_model} model...")
        self.gait_recognizer = GaitRecognizer(model_type=selected_model)
        
        # Show current model information
        model_info = self.gait_recognizer.get_model_info()
        vprint(f"Gait recognizer info: {model_info}")
        
        # Initialize face embedding extractor if enabled
        self._initialize_face_extractor()
        
    def _initialize_face_extractor(self):
        """Initialize face extractor if enabled"""
        if not getattr(self.config, 'ENABLE_FACE_RECOGNITION', True):
            return
            
        try:
            self.face_extractor = FaceEmbeddingExtractor(
                device=getattr(self.config, 'DEVICE', 'cuda:0'),
                det_name=getattr(self.config, 'FACE_DETECTION_MODEL', 'buffalo_l'),
                rec_name=getattr(self.config, 'FACE_RECOGNITION_MODEL', 'buffalo_l'),
                det_size=getattr(self.config, 'FACE_DETECTION_SIZE', (320, 320))
            )
            vprint("Face embedding extractor initialized successfully")
        except Exception as e:
            print(f"Warning: Face embedding extractor initialization failed: {e}")
            print("Continuing with gait-only recognition...")
            self.face_extractor = None