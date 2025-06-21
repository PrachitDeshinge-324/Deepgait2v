"""
XGait Model Accuracy Diagnostic and Improvement Tool

This script diagnoses potential issues causing low accuracy in XGait inference
and provides solutions for common problems.
"""

import numpy as np
import torch
import cv2
import os
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGaitDiagnostic:
    """Comprehensive diagnostic tool for XGait model inference issues"""
    
    def __init__(self, config):
        self.config = config
        self.issues_found = []
        self.recommendations = []
        
    def run_full_diagnostic(self, gait_recognizer, test_silhouettes=None, test_parsings=None):
        """Run comprehensive diagnostic on XGait model"""
        print("🔍 Starting XGait Model Diagnostic")
        print("=" * 60)
        
        # 1. Model configuration validation
        self._check_model_configuration(gait_recognizer)
        
        # 2. Input preprocessing validation
        self._check_input_preprocessing(gait_recognizer, test_silhouettes, test_parsings)
        
        # 3. Tensor format validation
        self._check_tensor_formats(gait_recognizer)
        
        # 4. Feature extraction validation
        self._check_feature_extraction(gait_recognizer)
        
        # 5. Parsing quality validation
        self._check_parsing_quality(gait_recognizer, test_silhouettes)
        
        # 6. Multimodal fusion validation
        self._check_multimodal_fusion()
        
        # 7. Domain mismatch analysis
        self._check_domain_mismatch()
        
        # Generate report
        self._generate_diagnostic_report()
        
        return self.issues_found, self.recommendations
    
    def _check_model_configuration(self, gait_recognizer):
        """Check model configuration issues"""
        print("📋 Checking Model Configuration...")
        
        # Check model type
        if gait_recognizer.model_type != "XGait":
            self.issues_found.append("❌ Model type mismatch - not using XGait")
            return
        
        # Check if XGait model loaded correctly
        if not hasattr(gait_recognizer, 'model') or gait_recognizer.model is None:
            self.issues_found.append("❌ XGait model not loaded correctly")
            self.recommendations.append("🔧 Check model weights path and loading process")
            return
        
        # Check XGait-specific components
        required_components = ['Backbone_sil', 'Backbone_par', 'gcm', 'pcm_up', 'pcm_middle', 'pcm_down']
        missing_components = []
        
        for component in required_components:
            if not hasattr(gait_recognizer.model, component):
                missing_components.append(component)
        
        if missing_components:
            self.issues_found.append(f"❌ Missing XGait components: {missing_components}")
            self.recommendations.append("🔧 Verify XGait model architecture and weights compatibility")
        else:
            print("✅ XGait model components loaded correctly")
    
    def _check_input_preprocessing(self, gait_recognizer, test_silhouettes, test_parsings):
        """Check input preprocessing pipeline"""
        print("🔄 Checking Input Preprocessing...")
        
        if test_silhouettes is None:
            test_silhouettes = self._create_test_silhouettes()
        
        try:
            # Test preprocessing
            preprocessed_input, seq_len = gait_recognizer.preprocess_silhouettes(
                test_silhouettes, test_parsings
            )
            
            if gait_recognizer.model_type == "XGait":
                pars_tensor, sils_tensor = preprocessed_input
                
                # Check tensor shapes
                expected_shape = (1, len(test_silhouettes), 64, 44)
                
                if pars_tensor.shape != expected_shape:
                    self.issues_found.append(f"❌ Parsing tensor shape mismatch: {pars_tensor.shape} vs {expected_shape}")
                
                if sils_tensor.shape != expected_shape:
                    self.issues_found.append(f"❌ Silhouette tensor shape mismatch: {sils_tensor.shape} vs {expected_shape}")
                
                # Check tensor value ranges
                if pars_tensor.min() < 0 or pars_tensor.max() > 1:
                    self.issues_found.append(f"❌ Parsing tensor values out of range [0,1]: [{pars_tensor.min():.3f}, {pars_tensor.max():.3f}]")
                
                if sils_tensor.min() < 0 or sils_tensor.max() > 1:
                    self.issues_found.append(f"❌ Silhouette tensor values out of range [0,1]: [{sils_tensor.min():.3f}, {sils_tensor.max():.3f}]")
                
                # Check if tensors contain meaningful data
                if torch.sum(pars_tensor) == 0:
                    self.issues_found.append("❌ Parsing tensor is empty (all zeros)")
                    self.recommendations.append("🔧 Check parsing generation - may need improved fallback")
                
                if torch.sum(sils_tensor) == 0:
                    self.issues_found.append("❌ Silhouette tensor is empty (all zeros)")
                    self.recommendations.append("🔧 Check silhouette extraction and preprocessing")
                
                print(f"✅ Preprocessing successful - shapes: pars={pars_tensor.shape}, sils={sils_tensor.shape}")
                
        except Exception as e:
            self.issues_found.append(f"❌ Preprocessing failed: {str(e)}")
            self.recommendations.append("🔧 Debug preprocessing pipeline for XGait compatibility")
    
    def _check_tensor_formats(self, gait_recognizer):
        """Check tensor format compatibility with XGait"""
        print("📐 Checking Tensor Format Compatibility...")
        
        # Create test data
        test_silhouettes = self._create_test_silhouettes()
        
        try:
            # Check both possible tensor formats that XGait might expect
            formats_to_test = [
                "standard",  # [batch, seq, h, w]
                "with_channel"  # [batch, seq, 1, h, w]
            ]
            
            for format_type in formats_to_test:
                try:
                    if format_type == "standard":
                        # Standard format
                        pars_tensor = torch.zeros((1, len(test_silhouettes), 64, 44))
                        sils_tensor = torch.zeros((1, len(test_silhouettes), 64, 44))
                    else:
                        # With channel dimension
                        pars_tensor = torch.zeros((1, len(test_silhouettes), 1, 64, 44))
                        sils_tensor = torch.zeros((1, len(test_silhouettes), 1, 64, 44))
                    
                    # Test if model accepts this format
                    with torch.no_grad():
                        # Create dummy inputs
                        labs = torch.zeros(1).long()
                        typs = torch.zeros(1).long()
                        vies = torch.zeros(1).long()
                        seqL = [torch.tensor([len(test_silhouettes)], dtype=torch.long)]
                        
                        inputs = ([pars_tensor, sils_tensor], labs, typs, vies, seqL)
                        
                        # Try forward pass (will fail but should give format insights)
                        try:
                            _ = gait_recognizer.model(inputs)
                            print(f"✅ {format_type} format accepted by model")
                        except Exception as format_error:
                            if "shape" in str(format_error).lower():
                                print(f"❌ {format_type} format rejected: {format_error}")
                            
                except Exception as e:
                    print(f"❌ Error testing {format_type} format: {e}")
                    
        except Exception as e:
            self.issues_found.append(f"❌ Tensor format testing failed: {str(e)}")
    
    def _check_feature_extraction(self, gait_recognizer):
        """Check feature extraction quality"""
        print("🎯 Checking Feature Extraction Quality...")
        
        test_silhouettes = self._create_test_silhouettes()
        
        try:
            # Test feature extraction
            embedding = gait_recognizer.recognize(test_silhouettes)
            
            if embedding is None:
                self.issues_found.append("❌ Feature extraction returned None")
                self.recommendations.append("🔧 Debug model forward pass and output processing")
                return
            
            # Check embedding properties
            if len(embedding.shape) != 2:
                self.issues_found.append(f"❌ Unexpected embedding shape: {embedding.shape}")
            
            # Check for NaN or infinite values
            if np.isnan(embedding).any():
                self.issues_found.append("❌ NaN values in embeddings")
                self.recommendations.append("🔧 Check model weights and input normalization")
            
            if np.isinf(embedding).any():
                self.issues_found.append("❌ Infinite values in embeddings")
                
            # Check embedding diversity (should not be all zeros or constant)
            if np.std(embedding) < 1e-6:
                self.issues_found.append("❌ Embeddings lack diversity (nearly constant)")
                self.recommendations.append("🔧 Model may not be properly trained or loaded")
            
            # Check embedding magnitude
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm < 1e-3:
                self.issues_found.append(f"❌ Embeddings have very small magnitude: {embedding_norm}")
            elif embedding_norm > 1e3:
                self.issues_found.append(f"❌ Embeddings have very large magnitude: {embedding_norm}")
            
            print(f"✅ Feature extraction successful - shape: {embedding.shape}, norm: {embedding_norm:.3f}")
            
        except Exception as e:
            self.issues_found.append(f"❌ Feature extraction failed: {str(e)}")
            self.recommendations.append("🔧 Debug XGait model forward pass")
    
    def _check_parsing_quality(self, gait_recognizer, test_silhouettes):
        """Check parsing map quality"""
        print("🗺️  Checking Parsing Map Quality...")
        
        if test_silhouettes is None:
            test_silhouettes = self._create_test_silhouettes()
        
        try:
            # Generate parsing maps
            parsing_maps = gait_recognizer.generate_parsing(test_silhouettes)
            
            for i, parsing_map in enumerate(parsing_maps):
                unique_labels = np.unique(parsing_map)
                
                # Check if parsing has meaningful segmentation
                if len(unique_labels) < 3:
                    self.issues_found.append(f"❌ Parsing map {i} has too few segments: {unique_labels}")
                    self.recommendations.append("🔧 Improve parsing generation - consider using deep learning models")
                
                # Check if background (0) dominates too much
                if 0 in unique_labels:
                    background_ratio = np.sum(parsing_map == 0) / parsing_map.size
                    if background_ratio > 0.8:
                        self.issues_found.append(f"❌ Parsing map {i} has {background_ratio:.1%} background")
                        self.recommendations.append("🔧 Improve silhouette quality or parsing algorithm")
                
                # Check for expected body parts
                expected_min_parts = 3  # At least head, torso, legs
                non_background_parts = len(unique_labels) - (1 if 0 in unique_labels else 0)
                if non_background_parts < expected_min_parts:
                    self.issues_found.append(f"❌ Insufficient body parts in parsing map {i}: {non_background_parts}")
            
            print(f"✅ Parsing generation successful - {len(parsing_maps)} maps with avg {np.mean([len(np.unique(p)) for p in parsing_maps]):.1f} segments")
            
        except Exception as e:
            self.issues_found.append(f"❌ Parsing generation failed: {str(e)}")
            self.recommendations.append("🔧 Fix parsing generation pipeline")
    
    def _check_multimodal_fusion(self):
        """Check multimodal fusion strategy"""
        print("🔗 Checking Multimodal Fusion Strategy...")
        
        # Check fusion weights
        face_weight = getattr(self.config, 'FACE_WEIGHT', 0.7)
        gait_weight = getattr(self.config, 'GAIT_WEIGHT', 0.3)
        
        if abs(face_weight + gait_weight - 1.0) > 1e-3:
            self.issues_found.append(f"❌ Fusion weights don't sum to 1.0: face={face_weight}, gait={gait_weight}")
            self.recommendations.append("🔧 Normalize fusion weights")
        
        # Check if face weight is too high for gait-focused system
        if face_weight > 0.8:
            self.issues_found.append(f"❌ Face weight too high ({face_weight}) - may overshadow gait features")
            self.recommendations.append("🔧 Consider reducing face weight to 0.5-0.6 for better balance")
        
        # Check adaptive weighting
        if not hasattr(self.config, 'ADAPTIVE_FUSION_WEIGHTING'):
            self.recommendations.append("🔧 Consider implementing adaptive fusion weighting based on feature quality")
        
        print(f"✅ Fusion weights: face={face_weight}, gait={gait_weight}")
    
    def _check_domain_mismatch(self):
        """Check for potential domain mismatch issues"""
        print("🎭 Checking Domain Mismatch Issues...")
        
        potential_mismatches = []
        
        # Check resolution mismatch
        if hasattr(self.config, 'INPUT_RESOLUTION'):
            if self.config.INPUT_RESOLUTION != (64, 44):
                potential_mismatches.append(f"Input resolution {self.config.INPUT_RESOLUTION} vs training (64, 44)")
        
        # Check dataset domain
        training_dataset = "Gait3D"  # Based on model name
        if hasattr(self.config, 'SOURCE_DATASET'):
            if self.config.SOURCE_DATASET.lower() != training_dataset.lower():
                potential_mismatches.append(f"Dataset mismatch: using {self.config.SOURCE_DATASET} vs trained on {training_dataset}")
        
        # Check viewing conditions
        potential_mismatches.extend([
            "Camera angle differences (XGait trained on controlled angles)",
            "Lighting conditions (indoor/outdoor variations)",
            "Walking surface (treadmill vs natural walking)",
            "Clothing variations (loose vs tight clothing)",
            "Walking speed variations",
            "Background complexity"
        ])
        
        if potential_mismatches:
            self.issues_found.extend([f"⚠️  Potential domain mismatch: {issue}" for issue in potential_mismatches])
            self.recommendations.extend([
                "🔧 Apply domain adaptation techniques",
                "🔧 Use data augmentation during inference",
                "🔧 Consider fine-tuning on target domain data",
                "🔧 Apply cross-camera normalization"
            ])
        
        print(f"⚠️  Found {len(potential_mismatches)} potential domain issues")
    
    def _create_test_silhouettes(self, num_frames=10):
        """Create test silhouettes for diagnostic"""
        test_silhouettes = []
        for i in range(num_frames):
            # Create a simple human-like silhouette
            sil = np.zeros((64, 44), dtype=np.uint8)
            
            # Head
            cv2.circle(sil, (22, 10), 5, 255, -1)
            
            # Body
            cv2.rectangle(sil, (15, 15), (29, 45), 255, -1)
            
            # Legs (with slight variation per frame for walking motion)
            leg_offset = int(2 * np.sin(i * 0.5))
            cv2.rectangle(sil, (18 + leg_offset, 45), (22, 60), 255, -1)  # Left leg
            cv2.rectangle(sil, (22 - leg_offset, 45), (26, 60), 255, -1)  # Right leg
            
            test_silhouettes.append(sil)
        
        return test_silhouettes
    
    def _generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC REPORT")
        print("=" * 60)
        
        if not self.issues_found:
            print("✅ No critical issues found!")
        else:
            print(f"❌ Found {len(self.issues_found)} issues:")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"  {i}. {issue}")
        
        print(f"\n🔧 {len(self.recommendations)} Recommendations:")
        for i, rec in enumerate(self.recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Priority recommendations for XGait
        print("\n🎯 PRIORITY FIXES FOR XGAIT:")
        priority_fixes = [
            "1. Verify XGait model weights are compatible with Gait3D architecture",
            "2. Ensure parsing maps have sufficient semantic detail (6+ body parts)",
            "3. Check input tensor format matches XGait expectations",
            "4. Validate silhouette quality and resolution (64x44)",
            "5. Consider domain adaptation for your specific camera/environment",
            "6. Balance face/gait fusion weights (try 0.4/0.6 or 0.5/0.5)",
            "7. Apply cross-camera normalization if using different cameras"
        ]
        
        for fix in priority_fixes:
            print(f"  {fix}")
