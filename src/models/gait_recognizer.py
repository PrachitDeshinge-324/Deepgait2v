import torch
import numpy as np
import sys
import os
import logging
from src.utils.device import get_best_device
import cv2

# Add OpenGait to path to import the model
sys.path.append(os.path.join(os.path.dirname(__file__), "../../OpenGait"))

# Import necessary modules for monkey patching
import torch.distributed as dist
from opengait.utils.msg_manager import MessageManager, get_msg_mgr
import opengait.modeling.base_model as base_model

# Save original methods to restore later if needed
original_get_loader = base_model.BaseModel.get_loader
original_init = base_model.BaseModel.__init__

# Create a bypass loader method that doesn't require dataset_root
def bypass_loader(self, data_cfg, train):
    print("Bypassing dataset loader for inference-only mode")
    # Return a dummy loader that won't be used for inference
    class DummyLoader:
        def __iter__(self): return self
        def __next__(self): raise StopIteration
    return DummyLoader()

# Create a patched __init__ method to handle Apple Silicon (MPS) devices
def patched_init(self, cfgs, training):
    """Patched initialization for BaseModel to support MPS devices"""
    super(base_model.BaseModel, self).__init__()
    self.msg_mgr = get_msg_mgr()
    self.cfgs = cfgs
    self.iteration = 0
    self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    if self.engine_cfg is None:
        raise Exception("Initialize a model without -Engine-Cfgs-")

    if training and self.engine_cfg['enable_float16']:
        from torch.cuda.amp import GradScaler
        self.Scaler = GradScaler()
        
    self.save_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                              cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

    self.build_network(cfgs['model_cfg'])
    self.init_parameters()
    
    # Check if this is training mode or not
    if 'trainer_cfg' in cfgs and 'transform' in cfgs['trainer_cfg']:
        from opengait.data import transform as transform
        self.trainer_trfs = transform.get_transform(cfgs['trainer_cfg']['transform'])
    else:
        self.trainer_trfs = []

    self.msg_mgr.log_info(cfgs['data_cfg'])
    if training:
        self.train_loader = self.get_loader(
            cfgs['data_cfg'], train=True)
    if not training or self.engine_cfg['with_test']:
        self.test_loader = self.get_loader(
            cfgs['data_cfg'], train=False)
        if 'transform' in self.engine_cfg:
            from opengait.data import transform as transform
            self.evaluator_trfs = transform.get_transform(
                self.engine_cfg['transform'])

    # Don't use CUDA-specific device setup - use the device from get_best_device()
    self.device = 0  # Just a placeholder for compatibility
    self.to(device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

    if training:
        from opengait.loss import LossAggregator
        self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
        self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
        self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
    self.train(training)
    restore_hint = self.engine_cfg['restore_hint']
    if restore_hint != 0:
        self.resume_ckpt(restore_hint)

# Apply the monkey patches
base_model.BaseModel.get_loader = bypass_loader
base_model.BaseModel.__init__ = patched_init

# Set up logging
def setup_simple_logger():
    logger = logging.getLogger("gait_inference")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Override logging and distributed functions
MessageManager.logger = setup_simple_logger()
MessageManager.log_info = lambda self, *args, **kwargs: print(f"INFO: {args[0] if args else kwargs.get('msg', '')}")
MessageManager.log_warning = lambda self, *args, **kwargs: print(f"WARNING: {args[0] if args else kwargs.get('msg', '')}")
dist.get_rank = lambda: 0
dist.get_world_size = lambda: 1
dist.is_initialized = lambda: True

# Now we can safely import all three models
from opengait.modeling.models import DeepGaitV2
from opengait.modeling.models.baseline import Baseline as GaitBase

# Import SkeletonGait++ with custom loader to resolve import issues
SkeletonGaitPP = None
try:
    from ..models.skeletongait_loader import SkeletonGaitPP, create_skeletongait_pp
    print("✓ SkeletonGaitPP imported successfully via custom loader")
except Exception as e:
    print(f"Info: SkeletonGait++ not available ({str(e)[:50]}...)")
    print("DeepGaitV2 and GaitBase models are fully functional.")
    SkeletonGaitPP = None

# Import XGait with custom loader
try:
    from opengait.modeling.models.xgait import XGait
    print("✓ XGait imported successfully")
except Exception as e:
    print(f"Info: XGait not available ({str(e)[:50]}...)")
    print("Please ensure XGait model is properly set up in OpenGait.")
    XGait = None

class GaitRecognizer:
    def __init__(self, model_type="DeepGaitV2", model_path=None, cfg=None):
        """
        Initialize the gait recognition model based on config settings
        
        Args:
            model_type (str): Type of model to use - will use config.GAIT_MODEL_TYPE if None
            model_path (str): Path to the trained model checkpoint - will use config path if None
            cfg (dict): Configuration for the model - will use config settings if None
        """
        # Use config values if not provided
        if model_type is None:
            from config import GAIT_MODEL_TYPE
            model_type = GAIT_MODEL_TYPE
            
        if model_path is None:
            from config import get_current_model_path
            model_path = get_current_model_path()
            
        if cfg is None:
            from config import get_current_model_config
            cfg = get_current_model_config()
        
        self.model_type = model_type
        self.current_model_type = model_type  # For compatibility with display code
        self.device = self._setup_device()
        
        # Validate model type
        if model_type not in ["DeepGaitV2", "GaitBase", "SkeletonGaitPP", "XGait"]:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'DeepGaitV2', 'GaitBase', 'SkeletonGaitPP', or 'XGait'")
        
        # Check if SkeletonGaitPP is available when requested
        if model_type == "SkeletonGaitPP" and SkeletonGaitPP is None:
            print("\n" + "="*60)
            print("⚠️  SKELETONGAIT++ NOT AVAILABLE")
            print("="*60)
            print("SkeletonGait++ model could not be imported.")
            print("Please check if OpenGait dependencies are properly installed.")
            print("\nRECOMMENDED ALTERNATIVES:")
            print("1. Use DeepGaitV2: Change config.GAIT_MODEL_TYPE = 'DeepGaitV2'")
            print("2. Use GaitBase: Change config.GAIT_MODEL_TYPE = 'GaitBase'")
            print("="*60)
            raise ValueError("SkeletonGaitPP not available. Please use DeepGaitV2 or GaitBase.")

        # Check if XGait is available when requested
        if model_type == "XGait" and XGait is None:
            print("\n" + "="*60)
            print("⚠️  XGAIT NOT AVAILABLE")
            print("="*60)
            print("XGait model could not be imported.")
            print("Please check if OpenGait dependencies are properly installed.")
            print("\nRECOMMENDED ALTERNATIVES:")
            print("1. Use DeepGaitV2: Change config.GAIT_MODEL_TYPE = 'DeepGaitV2'")
            print("2. Use GaitBase: Change config.GAIT_MODEL_TYPE = 'GaitBase'")
            print("="*60)
            raise ValueError("XGait not available. Please use DeepGaitV2 or GaitBase.")
        
        # Validate and adjust configuration for compatibility
        cfg = self._validate_config(model_type, cfg)
        
        # Create a complete config for the model
        complete_cfg = self._create_complete_config(model_type, cfg)
        
        # Initialize model
        print(f"Loading {model_type} model from {model_path}...")
        
        try:
            if model_type == "DeepGaitV2":
                self.model = DeepGaitV2(complete_cfg, False)
            elif model_type == "GaitBase":
                self.model = GaitBase(complete_cfg, False)
            elif model_type == "XGait":
                self.model = XGait(complete_cfg, False)
            else:  # SkeletonGaitPP
                self.model = SkeletonGaitPP(complete_cfg, False)
                
            self.model.to(self.device)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
                
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            print(f"{model_type} model loaded successfully")
            
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
        # Initialize improved inference for XGait
        if model_type == "XGait":
            try:
                from ..utils.improved_xgait_inference import ImprovedXGaitInference
                self.improved_inference = ImprovedXGaitInference(self, cfg or self._get_default_config())
                print("✅ Enhanced XGait inference pipeline initialized")
            except ImportError:
                print("⚠️ Enhanced XGait inference not available, using standard pipeline")
                self.improved_inference = None
        else:
            self.improved_inference = None
    
    def _setup_device(self):
        """Setup the appropriate device for inference"""
        device = get_best_device()
        if device == 'mps':
            # Force CPU for 3D operations which aren't supported on MPS
            actual_device = torch.device("cpu")
            print(f"Using CPU for model inference (3D convolutions not supported on MPS)")
        else:
            actual_device = torch.device(device)
        return actual_device
    
    def _validate_config(self, model_type, cfg):
        """
        Validate and adjust configuration for compatibility
        
        Args:
            model_type: Type of model
            cfg: Configuration dictionary
            
        Returns:
            Validated configuration
        """
        cfg_copy = cfg.copy()
        
        if model_type == "DeepGaitV2":
            # Check for 3D convolution modes that might not be supported
            if 'Backbone' in cfg_copy and 'mode' in cfg_copy['Backbone']:
                if cfg_copy['Backbone']['mode'] == 'p3d':
                    print("WARNING: p3d mode detected. This uses 3D convolutions.")
                    print("Switching to 2d mode for better compatibility...")
                    cfg_copy['Backbone']['mode'] = '2d'
                    
            # Ensure required fields are present
            if 'SeparateBNNecks' not in cfg_copy:
                cfg_copy['SeparateBNNecks'] = {'class_num': 3000}
                
        elif model_type == "SkeletonGaitPP":
            # SkeletonGait++ requires specific configuration
            if 'Backbone' not in cfg_copy:
                cfg_copy['Backbone'] = {
                    'in_channels': 3,
                    'blocks': [1, 4, 4, 1],
                    'C': 2
                }
            
            # Ensure required fields are present
            if 'SeparateBNNecks' not in cfg_copy:
                cfg_copy['SeparateBNNecks'] = {'class_num': 3000}
                
            # Set input channels for multi-modal input (pose + silhouette)
            if cfg_copy['Backbone']['in_channels'] != 3:
                print("WARNING: SkeletonGait++ requires 3 input channels (2 for pose + 1 for silhouette)")
                cfg_copy['Backbone']['in_channels'] = 3
                
        elif model_type == "GaitBase":
            # Validate GaitBase specific configurations
            required_fields = ['backbone_cfg', 'SeparateFCs', 'SeparateBNNecks', 'bin_num']
            for field in required_fields:
                if field not in cfg_copy:
                    print(f"WARNING: Missing required field '{field}' in GaitBase config")
                    
            # Ensure bin_num is present
            if 'bin_num' not in cfg_copy:
                cfg_copy['bin_num'] = [16]
                
        elif model_type == "XGait":
            # XGait requires specific configuration
            required_fields = ['backbone_cfg', 'CALayers', 'CALayersP', 'SeparateFCs', 'SeparateBNNecks', 'bin_num']
            for field in required_fields:
                if field not in cfg_copy:
                    print(f"WARNING: Missing required field '{field}' in XGait config")
            
            # Ensure bin_num is present
            if 'bin_num' not in cfg_copy:
                cfg_copy['bin_num'] = [16]
                
        return cfg_copy
    
    def get_model_info(self):
        """Get information about current model"""
        model_info = {
            'model_type': self.model_type,
            'device': str(self.device)
        }
        return model_info

    def _create_complete_config(self, model_type, cfg):
        """Create complete configuration for the model"""
        if model_type == "DeepGaitV2":
            model_type_name = "DeepGaitV2"
        elif model_type == "SkeletonGaitPP":
            model_type_name = "SkeletonGaitPP"
        elif model_type == "XGait":
            model_type_name = "XGait"
        else:  # GaitBase
            model_type_name = "Baseline"
        
        complete_cfg = {
            'model_cfg': {
                'model': model_type_name,
                **cfg
            },
            'data_cfg': {
                'dataset_name': 'inference',
                'num_workers': 0,
                'dataset_root': 'dummy_path',
                'dataset_partition': 'dummy_partition.json'
            },
            'trainer_cfg': {
                'transform': [],
                'enable_float16': False,
                'save_name': 'inference',
                'with_test': False,
                'sampler': {
                    'type': 'InferenceSampler',
                    'batch_size': 1
                }
            },
            'evaluator_cfg': {
                'transform': [],
                'sampler': {
                    'type': 'InferenceSampler',
                    'batch_size': 1
                },
                'restore_ckpt_strict': False,
                'restore_hint': 0,
                'save_name': 'inference',
                'enable_float16': False
            }
        }
        return complete_cfg

    def preprocess_silhouettes(self, silhouettes, parsings=None):
        """
        Preprocess silhouettes to match model training pipeline
        DeepGaitV2 and GaitBase use silhouettes only
        SkeletonGaitPP uses pose+silhouette
        XGait uses silhouette+parsing
        
        Args:
            silhouettes: List of silhouette arrays
            parsings: List of parsing arrays (for XGait)
        """
        seq_len = len(silhouettes)
        
        # Resize silhouettes to expected dimensions (height=64, width=44)
        resized_sils = []
        for sil in silhouettes:
            # Convert to grayscale if RGB
            if len(sil.shape) == 3 and sil.shape[2] == 3:
                sil = cv2.cvtColor(sil, cv2.COLOR_RGB2GRAY)
            elif len(sil.shape) == 3 and sil.shape[2] == 1:
                sil = sil.squeeze(2)
            
            # Resize to 64x44 - common gait silhouette size
            resized = cv2.resize(sil, (44, 64), interpolation=cv2.INTER_LINEAR)
            resized_sils.append(resized)
        
        if self.model_type == "XGait":
            # XGait requires both silhouette and parsing
            if parsings is None:
                # Use improved fallback: generate parsing maps from silhouettes
                resized_pars = self.generate_parsing(resized_sils)
            else:
                # Resize parsing maps
                resized_pars = []
                for par in parsings:
                    resized = cv2.resize(par, (44, 64), interpolation=cv2.INTER_NEAREST)
                    resized_pars.append(resized)
            
            # Prepare tensors for XGait without extra dimension
            sils_tensor = torch.zeros((1, seq_len, 64, 44), dtype=torch.float32)
            pars_tensor = torch.zeros((1, seq_len, 64, 44), dtype=torch.float32)
            
            for i, (sil, par) in enumerate(zip(resized_sils, resized_pars)):
                sils_tensor[0, i] = torch.from_numpy(sil).float() / 255.0
                pars_tensor[0, i] = torch.from_numpy(par).float() / 255.0
                
            return (pars_tensor, sils_tensor), seq_len

        elif self.model_type == "SkeletonGaitPP":
            # SkeletonGaitPP requires multimodal input: pose heatmaps (2 channels) + silhouette (1 channel)
            try:
                # Import pose generator with correct path
                from ..processing.pose_generator import PoseHeatmapGenerator
                pose_generator = PoseHeatmapGenerator(target_size=(64, 44))
                
                # Generate pose heatmaps from silhouettes (2 channels)
                pose_heatmaps_list = []
                sil_list = []
                
                for sil in resized_sils:
                    # Generate pose heatmap (2 channels)
                    pose_heatmap = pose_generator.generate_pose_from_silhouette(sil)
                    pose_heatmaps_list.append(pose_heatmap)
                    
                    # Keep silhouette as single channel
                    sil_normalized = sil.astype(np.float32) / 255.0
                    sil_list.append(sil_normalized)
                
                # Stack to create sequences
                pose_heatmaps = np.stack(pose_heatmaps_list, axis=0)  # [T, 2, H, W]
                silhouettes_seq = np.stack(sil_list, axis=0)  # [T, H, W]
                
                # The model expects the input to be concatenated in the preprocessing phase
                # but internally splits it into pose (first 2 channels) and silhouette (last channel)
                multimodal_input = np.concatenate([
                    pose_heatmaps,  # [T, 2, H, W]
                    silhouettes_seq[:, np.newaxis, :, :]  # [T, 1, H, W]
                ], axis=1)  # [T, 3, H, W]
                
                # Convert to tensor: [batch, sequence, channels, height, width]
                multimodal_tensor = torch.from_numpy(multimodal_input).float().unsqueeze(0)
                return multimodal_tensor, seq_len
                
            except ImportError:
                print("Warning: PoseHeatmapGenerator not available. Using silhouette-only input for SkeletonGaitPP.")
                # Fallback: create 3-channel input with silhouette duplicated
                sils_tensor = torch.zeros((1, seq_len, 3, 64, 44), dtype=torch.float32)
                for i, sil in enumerate(resized_sils):
                    sil_normalized = torch.from_numpy(sil).float() / 255.0
                    # Use silhouette for all 3 channels as fallback
                    sils_tensor[0, i, 0] = sil_normalized  # pose_x (placeholder)
                    sils_tensor[0, i, 1] = sil_normalized  # pose_y (placeholder)
                    sils_tensor[0, i, 2] = sil_normalized  # silhouette
                return sils_tensor, seq_len
        else:
            # DeepGaitV2 and GaitBase use single-channel silhouette input
            sils_tensor = torch.zeros((1, seq_len, 64, 44), dtype=torch.float32)
            for i, sil in enumerate(resized_sils):
                sils_tensor[0, i] = torch.from_numpy(sil).float() / 255.0
            return sils_tensor, seq_len

    def recognize(self, silhouettes, parsings=None):
        """
        Perform gait recognition using the currently active model
        
        Args:
            silhouettes: List of silhouette arrays
            parsings: List of parsing arrays (for XGait)
            
        Returns:
            Feature embeddings from the model
        """
        if len(silhouettes) == 0:
            print("No silhouettes provided")
            return None
        
        # Use improved inference for XGait if available
        if self.model_type == "XGait" and hasattr(self, 'improved_inference') and self.improved_inference is not None:
            try:
                return self.improved_inference.enhanced_inference(silhouettes, parsings)
            except Exception as e:
                print(f"Enhanced inference failed, falling back to standard: {e}")
        
        # Standard preprocessing and inference
        preprocessed_input, seq_len = self.preprocess_silhouettes(silhouettes, parsings)
        
        # Move to device
        if self.model_type == "XGait":
            pars_tensor, sils_tensor = preprocessed_input
            pars_tensor = pars_tensor.to(self.device)
            sils_tensor = sils_tensor.to(self.device)
        else:
            sils_tensor = preprocessed_input.to(self.device)
        
        # Create dummy values for unused inputs
        labs = torch.zeros(1).long().to(self.device)
        typs = torch.zeros(1).long().to(self.device) 
        vies = torch.zeros(1).long().to(self.device)
        
        # All models expect seqL as a list of sequence lengths
        seq_tensor = torch.tensor([seq_len], dtype=torch.long).to(self.device)
        seqL = [seq_tensor]
        
        # Run inference
        with torch.no_grad():
            try:
                if self.model_type == "XGait":
                    # XGait expects [parsing, silhouette] inputs
                    inputs = ([pars_tensor, sils_tensor], labs, typs, vies, seqL)
                elif self.model_type == "SkeletonGaitPP":
                    # SkeletonGaitPP expects multimodal input [pose_heatmaps, silhouettes]
                    # sils_tensor already contains [pose_x, pose_y, silhouette] in 3 channels
                    if sils_tensor.dim() == 5:  # [batch, seq, channels, height, width]
                        pose_heatmaps = sils_tensor[:, :, :2, :, :]  # First 2 channels for pose
                        silhouettes = sils_tensor[:, :, 2:3, :, :]   # Last channel for silhouette
                        # Combine pose channels into single heatmap input
                        pose_heatmaps = pose_heatmaps.view(pose_heatmaps.shape[0], pose_heatmaps.shape[1], -1, pose_heatmaps.shape[3], pose_heatmaps.shape[4])
                        inputs = ([pose_heatmaps, silhouettes.squeeze(2)], labs, typs, vies, seqL)
                    else:
                        # Fallback for unexpected tensor format
                        inputs = ([sils_tensor], labs, typs, vies, seqL)
                else:
                    # DeepGaitV2 and GaitBase use single silhouette input
                    inputs = ([sils_tensor], labs, typs, vies, seqL)
                    
                outputs = self.model(inputs)
                embeddings = outputs['inference_feat']['embeddings']
                return embeddings.cpu().numpy()
            except Exception as e:
                print(f"Error during inference with {self.model_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # If it's XGait with shape error, try alternative format
                if self.model_type == "XGait" and "shape" in str(e).lower():
                    try:
                        print("Attempting with alternative tensor format...")
                        # Try reshaping the tensors
                        pars_tensor_reshaped = pars_tensor.unsqueeze(2)  # [1, seq, 1, 64, 44]
                        sils_tensor_reshaped = sils_tensor.unsqueeze(2)  # [1, seq, 1, 64, 44]
                        inputs = ([pars_tensor_reshaped, sils_tensor_reshaped], labs, typs, vies, seqL)
                        
                        outputs = self.model(inputs)
                        embeddings = outputs['inference_feat']['embeddings']
                        print("Alternative tensor format successful!")
                        return embeddings.cpu().numpy()
                    except Exception as e2:
                        print(f"Alternative format also failed: {str(e2)}")
                
                return None

    def generate_parsing(self, silhouettes):
        """
        Generate parsing maps from silhouettes using state-of-the-art human parsing models
        
        Args:
            silhouettes: List of silhouette arrays (H x W, values 0-255)
        Returns:
            parsing_maps: List of parsing maps with semantic body part labels
        """
        # Initialize human parsing generator if not already done
        if not hasattr(self, '_parsing_generator'):
            try:
                from ..processing.human_parsing import create_human_parser
                
                # Try different models in order of preference
                models_to_try = ['schp_resnet101', 'lip_hrnet', 'atr_hrnet']
                
                for model_name in models_to_try:
                    try:
                        print(f"Initializing human parsing model: {model_name}")
                        self._parsing_generator = create_human_parser(
                            model_name=model_name, 
                            device=str(self.device)
                        )
                        print(f"Successfully loaded {model_name} for human parsing")
                        break
                    except Exception as e:
                        print(f"Failed to load {model_name}: {e}")
                        continue
                else:
                    # All models failed, use geometric fallback
                    print("All parsing models failed, using geometric fallback")
                    self._parsing_generator = None
                    
            except ImportError as e:
                print(f"Failed to import human parsing module: {e}")
                self._parsing_generator = None
        
        # Generate parsing maps
        if self._parsing_generator is not None:
            try:
                parsing_maps = self._parsing_generator.generate_from_silhouettes(silhouettes)
                print(f"Generated {len(parsing_maps)} parsing maps using advanced model")
                return parsing_maps
            except Exception as e:
                print(f"Human parsing model failed: {e}")
                # Fall through to geometric fallback
        
        # Geometric fallback implementation
        print("Using geometric fallback for parsing generation")
        parsing_maps = []
        for sil in silhouettes:
            h, w = sil.shape
            parsing_map = np.zeros_like(sil, dtype=np.uint8)
            
            # Find bounding box of the silhouette
            coords = np.column_stack(np.where(sil > 0))
            if coords.size == 0:
                parsing_maps.append(parsing_map)
                continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            height = y_max - y_min + 1
            
            # Enhanced geometric parsing with more body parts
            head_end = y_min + int(0.15 * height)      # Head: top 15%
            neck_end = y_min + int(0.20 * height)      # Neck: 15-20%
            torso_end = y_min + int(0.55 * height)     # Torso: 20-55%
            thigh_end = y_min + int(0.80 * height)     # Thighs: 55-80%
            # Legs: 80-100%
            
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if sil[y, x] > 0:
                        width_pos = (x - x_min) / max(1, x_max - x_min)
                        
                        if y < head_end:
                            parsing_map[y, x] = 1  # Head
                        elif y < neck_end:
                            parsing_map[y, x] = 2  # Neck/Upper torso
                        elif y < torso_end:
                            # Torso with arms detection
                            if width_pos < 0.25 or width_pos > 0.75:
                                parsing_map[y, x] = 3  # Arms (outer parts)
                            else:
                                parsing_map[y, x] = 2  # Torso (center)
                        elif y < thigh_end:
                            parsing_map[y, x] = 4  # Thighs/Upper legs
                        else:
                            parsing_map[y, x] = 5  # Lower legs/feet
            
            parsing_maps.append(parsing_map)
        
        return parsing_maps