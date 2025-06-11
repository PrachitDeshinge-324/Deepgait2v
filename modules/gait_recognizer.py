import torch
import numpy as np
import sys
import os
import logging
from utils.device import get_best_device
import cv2

# Add OpenGait to path to import the model
sys.path.append(os.path.join(os.path.dirname(__file__), "../OpenGait"))

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

# Now we can safely import DeepGaitV2
from opengait.modeling.models import DeepGaitV2

class GaitRecognizer:
    def __init__(self, model_path, cfg):
        """
        Initialize the gait recognition model
        
        Args:
            model_path (str): Path to the trained DeepGaitV2 model checkpoint
            cfg (dict): Configuration for the model
        """
        # Force CPU for 3D operations
        device = get_best_device()
        if device == 'mps':
            self.device = torch.device("cpu")
            print(f"Using CPU for model inference (3D convolutions not supported on MPS)")
        else:
            self.device = torch.device(device)
        # Check if we can switch to 2D mode instead of p3d
        if 'Backbone' in cfg and 'mode' in cfg['Backbone'] and cfg['Backbone']['mode'] == 'p3d':
            print("Detected p3d mode which uses 3D convolutions. Trying to switch to 2d mode...")
            try:
                # Try to modify the config to use 2D mode
                cfg['Backbone']['mode'] = '2d'
                print("Switched to 2d mode for compatibility with Apple Silicon")
            except Exception as e:
                print(f"Could not modify mode, continuing with CPU: {e}")
        
        # Create a complete config for the model
        complete_cfg = {
            'model_cfg': {
                'model': 'DeepGaitV2',
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
        
        # Initialize model
        print("Loading DeepGaitV2 model...")
        
        try:
            self.model = DeepGaitV2(complete_cfg, False)
            self.model.to(self.device)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
                
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            print("DeepGaitV2 model loaded successfully")
        except Exception as e:
            print(f"Error loading DeepGaitV2 model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def preprocess_silhouettes(self, silhouettes):
        """
        Preprocess silhouettes to match DeepGaitV2 training pipeline
        """
        seq_len = len(silhouettes)
        
        # Debug print for input verification
        print(f"Input: {len(silhouettes)} silhouettes, first shape: {silhouettes[0].shape}")
        
        # Resize silhouettes to expected dimensions (height=64, width=44)
        resized_sils = []
        for sil in silhouettes:
            # Resize to 64x44 - common gait silhouette size
            resized = cv2.resize(sil, (44, 64), interpolation=cv2.INTER_LINEAR)
            resized_sils.append(resized)
        
        # Create tensor with proper dimensions for DeepGaitV2
        # Shape should be [batch, channel, sequence, height, width]
        sils_tensor = torch.zeros((1, 1, seq_len, 64, 44), dtype=torch.float32)
        
        # Fill tensor with normalized silhouettes (0-1 range)
        for i, sil in enumerate(resized_sils):
            sils_tensor[0, 0, i] = torch.from_numpy(sil).float() / 255.0
        
        # Debug print for output verification
        print(f"Output tensor shape: {sils_tensor.shape}")
        
        return sils_tensor, seq_len

    def recognize(self, silhouettes):
        if len(silhouettes) == 0:
            print("No silhouettes provided")
            return None
            
        # Preprocess silhouettes
        sils_tensor, seq_len = self.preprocess_silhouettes(silhouettes)
        
        # Add debug print to check sils_tensor shape
        print(f"Preprocessed silhouettes shape: {sils_tensor.shape}, seq_len: {seq_len}")
        
        # Move to device
        sils_tensor = sils_tensor.to(self.device)
        
        # Create dummy values for unused inputs
        labs = torch.zeros(1).long().to(self.device)
        typs = torch.zeros(1).long().to(self.device) 
        vies = torch.zeros(1).long().to(self.device)
        
        # Fix: The TP module in DeepGaitV2 expects seqL as a list of sequence lengths
        # Make sure the tensor is constructed properly
        seq_tensor = torch.tensor([seq_len], dtype=torch.long).to(self.device)
        seqL = [seq_tensor]  # Wrap in list as expected by TP module
        
        # Debug print to verify seqL format
        print(f"seqL format: {type(seqL)}, value: {seqL}, element type: {type(seqL[0])}")
        
        # Run inference
        with torch.no_grad():
            try:
                inputs = (sils_tensor, labs, typs, vies, seqL)
                outputs = self.model(inputs)
                embeddings = outputs['inference_feat']['embeddings']
                return embeddings.cpu().numpy()
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                import traceback
                traceback.print_exc()
                return None