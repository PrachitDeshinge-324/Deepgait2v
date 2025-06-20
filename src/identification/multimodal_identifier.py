"""
Multi-Modal Person Identification for Enhanced CCTV Recognition
Combines face and gait biometrics for improved accuracy with cross-camera support
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from ..utils.database import PersonEmbeddingDatabase
from ..adapters.cross_camera_adapter import apply_cross_camera_adaptation, compute_cross_camera_similarity, domain_adapter
import config

class MultiModalIdentifier:
    """Enhanced identification system using both face and gait biometrics with cross-camera support"""
    
    def __init__(self, database_handler):
        """
        Initialize with database handler
        
        Args:
            database_handler: Database handler with both face and gait identification methods
        """
        self.database = database_handler
        self.sequence_history = {}  # For temporal consistency checks
        self.face_weight = getattr(config, 'FACE_WEIGHT', 0.7)  # Weight for face recognition
        self.gait_weight = getattr(config, 'GAIT_WEIGHT', 0.3)  # Weight for gait recognition
        self.require_both = getattr(config, 'REQUIRE_BOTH_MODALITIES', False)
        
        # Cross-camera adaptation settings
        self.enable_cross_camera = getattr(config, 'ENABLE_CROSS_CAMERA_ADAPTATION', True)
        self.camera_adaptive_similarity = getattr(config, 'CAMERA_ADAPTIVE_SIMILARITY', True)
        self.current_camera_id = "default"  # Current camera identifier
    
    def set_camera_id(self, camera_id: str):
        """Set the current camera identifier for domain adaptation"""
        self.current_camera_id = camera_id
    
    def collect_embeddings_for_adaptation(self, gait_embedding: Optional[np.ndarray] = None,
                                        face_embedding: Optional[np.ndarray] = None):
        """
        Collect embeddings for cross-camera domain adaptation
        
        Args:
            gait_embedding: Gait embedding to collect
            face_embedding: Face embedding to collect
        """
        if self.enable_cross_camera:
            embeddings = {}
            if gait_embedding is not None:
                embeddings['gait'] = gait_embedding
            if face_embedding is not None:
                embeddings['face'] = face_embedding
            
            if embeddings:
                domain_adapter.collect_camera_statistics(embeddings, self.current_camera_id)
    
    def fit_cross_camera_adaptation(self):
        """Fit the cross-camera domain adaptation models"""
        if self.enable_cross_camera:
            domain_adapter.fit_domain_adaptation()
    
    def identify_with_proximity_awareness(self, track_id: str, 
                                        gait_embedding: Optional[np.ndarray] = None,
                                        face_embedding: Optional[np.ndarray] = None, 
                                        quality: float = 0.5,
                                        is_close_to_camera: bool = False) -> List[Tuple]:
        """
        Identify person with awareness of proximity to camera and cross-camera adaptation
        
        Args:
            track_id: Track identifier
            gait_embedding: Gait embedding or None
            face_embedding: Face embedding or None
            quality: Overall sequence quality
            is_close_to_camera: Whether subject is close to camera
            
        Returns:
            List of (person_id, confidence, name, quality) tuples
        """
        # Apply cross-camera domain adaptation first
        if self.enable_cross_camera:
            # Collect embeddings for adaptation (in background)
            self.collect_embeddings_for_adaptation(gait_embedding, face_embedding)
            
            # Apply adaptation to current embeddings
            gait_embedding, face_embedding = apply_cross_camera_adaptation(
                gait_embedding, face_embedding, self.current_camera_id
            )
        # Store previous identification for this track
        previous_id = None
        previous_confidence = 0
        if track_id in self.sequence_history and self.sequence_history[track_id]:
            last_match = self.sequence_history[track_id][-1].get('best_match')
            if last_match:
                previous_id = last_match[0]
                previous_confidence = last_match[1]
        
        # Get new matches
        current_matches = self.identify_person(track_id, gait_embedding, face_embedding, quality)
        
        # If close to camera, trust current identification more
        # This would be a high-confidence identification
        if is_close_to_camera and face_embedding is not None and current_matches:
            self._update_tracking_history(track_id, gait_embedding, face_embedding, 
                                        current_matches, quality)
            return current_matches
        
        # If far from camera but previous identification exists, use stricter threshold
        # for new identifications to override previous one
        if previous_id and not is_close_to_camera and current_matches:
            best_match = current_matches[0]
            
            # Higher threshold for changing identity when far from camera
            # Only accept new identity if substantially more confident
            change_identity_threshold = getattr(config, 'DISTANT_IDENTITY_CHANGE_THRESHOLD', 0.3)
            
            # Check if new best match is different from previous ID
            if best_match[0] != previous_id:
                # Only change identity if the new match has significantly higher confidence
                if best_match[1] > previous_confidence + change_identity_threshold:
                    # Accept the new higher-confidence identity
                    self._update_tracking_history(track_id, gait_embedding, face_embedding, 
                                               current_matches, quality)
                    return current_matches
                else:
                    # Retain previous identity with slightly updated confidence
                    # Find previous identity in current matches if it exists
                    for match in current_matches:
                        if match[0] == previous_id:
                            # Found previous ID in current matches - use this
                            self._update_tracking_history(track_id, gait_embedding, face_embedding, 
                                                       [match], quality)
                            return [match]
                    
                    # Previous ID not in current matches, but stick with it
                    # Find the full details of previous ID from history
                    for history_item in reversed(self.sequence_history[track_id]):
                        if history_item.get('best_match') and history_item['best_match'][0] == previous_id:
                            prev_match = history_item['best_match']
                            # Slightly penalize confidence for persistent identity
                            adjusted_match = (prev_match[0], prev_match[1] * 0.95, 
                                            prev_match[2], prev_match[3])
                            return [adjusted_match]
            
        # Default: use current matches
        self._update_tracking_history(track_id, gait_embedding, face_embedding, 
                                    current_matches, quality)
        return current_matches
    
    def identify_person(self, track_id: str, gait_embedding: Optional[np.ndarray] = None,
                      face_embedding: Optional[np.ndarray] = None, quality: float = 0.5) -> List[Tuple]:
        """
        Identify person using available biometrics
        
        Args:
            track_id: Track identifier
            gait_embedding: Gait embedding or None
            face_embedding: Face embedding or None
            quality: Overall sequence quality
            
        Returns:
            List of (person_id, confidence, name, quality) tuples
        """
        face_matches = []
        gait_matches = []
        
        # Get matches from face if available
        if face_embedding is not None:
            face_matches = self.database.identify_person_face(
                face_embedding,
                top_k=5,
                threshold=getattr(config, 'FACE_THRESHOLD', 0.6)
            )
        
        # Get matches from gait if available
        if gait_embedding is not None:
            # Use the enhanced nucleus sampling from the existing system
            gait_matches = self.database.identify_person_nucleus(
                gait_embedding,
                top_p=getattr(config, 'NUCLEUS_TOP_P', 0.85),
                min_candidates=1,
                max_candidates=5,
                close_sim_threshold=getattr(config, 'NUCLEUS_CLOSE_SIM_THRESHOLD', 0.08),
                amplification_factor=getattr(config, 'NUCLEUS_AMPLIFICATION_FACTOR', 35.0),
                quality_weight=getattr(config, 'NUCLEUS_QUALITY_WEIGHT', 0.8),
                enhanced_ranking=getattr(config, 'NUCLEUS_ENHANCED_RANKING', True)
            )
        
        # Handle different scenarios
        if self.require_both and not (face_matches and gait_matches):
            return []  # Require both modalities
        elif face_matches and gait_matches:
            return self._fuse_multimodal_results(face_matches, gait_matches)
        elif face_matches:
            return face_matches
        elif gait_matches:
            return gait_matches
        else:
            return []
    
    def _fuse_multimodal_results(self, face_matches: List[Tuple], 
                               gait_matches: List[Tuple]) -> List[Tuple]:
        """
        Fuse face and gait recognition results with adaptive weighting
        
        Args:
            face_matches: List of face recognition matches
            gait_matches: List of gait recognition matches
            
        Returns:
            Fused list of matches
        """
        # Collect all person IDs
        all_person_ids = set()
        for matches in [face_matches, gait_matches]:
            for match in matches:
                all_person_ids.add(match[0])
        
        # Adaptive weight calculation based on match quality
        face_quality = np.mean([match[1] for match in face_matches]) if face_matches else 0.0
        gait_quality = np.mean([match[1] for match in gait_matches]) if gait_matches else 0.0
        
        # Adjust weights based on relative quality
        if face_quality > 0 and gait_quality > 0:
            total_quality = face_quality + gait_quality
            adaptive_face_weight = (face_quality / total_quality) * 0.6 + 0.4 * self.face_weight
            adaptive_gait_weight = (gait_quality / total_quality) * 0.6 + 0.4 * self.gait_weight
            # Normalize
            total_weight = adaptive_face_weight + adaptive_gait_weight
            adaptive_face_weight /= total_weight
            adaptive_gait_weight /= total_weight
        else:
            adaptive_face_weight = self.face_weight
            adaptive_gait_weight = self.gait_weight
        
        # Calculate fused scores
        fused_results = []
        for person_id in all_person_ids:
            face_score = 0.0
            gait_score = 0.0
            person_name = None
            person_quality = 0.0
            
            # Get face score if available
            for match in face_matches:
                if match[0] == person_id:
                    face_score = match[1]
                    person_name = match[2]
                    person_quality = match[3]
                    break
            
            # Get gait score if available
            for match in gait_matches:
                if match[0] == person_id:
                    gait_score = match[1]
                    if not person_name:
                        person_name = match[2]
                        person_quality = match[3]
                    break
            
            # Calculate weighted score using adaptive weights
            fused_score = (adaptive_face_weight * face_score) + (adaptive_gait_weight * gait_score)
            
            if person_name:
                fused_results.append((person_id, fused_score, person_name, person_quality))
        
        # Sort by fused score and return top candidates
        fused_results.sort(key=lambda x: x[1], reverse=True)
        max_candidates = getattr(config, 'NUCLEUS_MAX_CANDIDATES', 5)
        return fused_results[:max_candidates]
    
    def enhanced_identification(self, track_id: str, gait_embedding: Optional[np.ndarray] = None,
                              face_embedding: Optional[np.ndarray] = None, 
                              sequence_quality: float = 0.5) -> List[Tuple]:
        """
        Enhanced identification with temporal consistency and ensemble methods
        
        Args:
            track_id: Tracking ID
            gait_embedding: Gait embedding (optional)
            face_embedding: Face embedding (optional)
            sequence_quality: Quality score of the sequence
            
        Returns:
            List of (person_id, confidence, name, quality) tuples
        """
        
        # Strategy 1: Direct multimodal identification
        direct_matches = self.identify_person(track_id, gait_embedding, face_embedding, sequence_quality)
        
        # Strategy 2: Temporal consistency check
        temporal_matches = self._temporal_consistency_check(track_id, gait_embedding, face_embedding)
        
        # Strategy 3: Quality-weighted fusion of strategies
        final_matches = self._fuse_identification_strategies(
            direct_matches, temporal_matches, sequence_quality
        )
        
        # Update tracking history
        self._update_tracking_history(track_id, gait_embedding, face_embedding, final_matches, sequence_quality)
        
        return final_matches
    
    def _temporal_consistency_check(self, track_id: str, gait_embedding: Optional[np.ndarray] = None,
                                   face_embedding: Optional[np.ndarray] = None) -> List[Tuple]:
        """Check consistency with previous identifications for this track"""
        if track_id not in self.sequence_history:
            return []
        
        history = self.sequence_history[track_id]
        if len(history) < 2:
            return []
        
        # Get recent identities (last 3-5 sequences)
        recent_identities = [h['best_match'] for h in history[-3:] if h['best_match']]
        
        if not recent_identities:
            return []
        
        # Check if there's a consistent identity
        identity_counts = {}
        for identity in recent_identities:
            person_id = identity[0]  # Extract person_id from (person_id, similarity, name, quality)
            identity_counts[person_id] = identity_counts.get(person_id, 0) + 1
        
        # Return the most consistent identity with boosted confidence
        if identity_counts:
            most_consistent = max(identity_counts, key=identity_counts.get)
            consistency_boost = identity_counts[most_consistent] / len(recent_identities)
            
            # Get the latest match for this person
            for identity in reversed(recent_identities):
                if identity[0] == most_consistent:
                    return [(identity[0], identity[1] * (1 + consistency_boost * 0.3), 
                           identity[2], identity[3])]
        
        return []
    
    def _fuse_identification_strategies(self, direct: List, temporal: List, 
                                      quality: float) -> List[Tuple]:
        """Fuse results from different identification strategies"""
        
        # Weight different strategies
        weights = {
            'direct': 0.7,
            'temporal': 0.3 if temporal else 0.0
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Collect all person IDs mentioned
        all_person_ids = set()
        for matches in [direct, temporal]:
            for match in matches:
                if match:  # Check if match is not None/empty
                    all_person_ids.add(match[0])
        
        # Calculate fused scores for each person
        fused_results = []
        for person_id in all_person_ids:
            fused_score = 0.0
            person_name = None
            person_quality = 0.0
            
            # Direct matches
            for match in direct:
                if match[0] == person_id:
                    fused_score += weights['direct'] * match[1]
                    person_name = match[2]
                    person_quality = match[3]
                    break
            
            # Temporal matches
            for match in temporal:
                if match[0] == person_id:
                    fused_score += weights['temporal'] * match[1]
                    if not person_name:
                        person_name = match[2]
                        person_quality = match[3]
                    break
            
            # Apply quality boost
            quality_boost = 1.0 + (quality - 0.5) * 0.2  # Boost for high-quality sequences
            fused_score *= quality_boost
            
            if person_name:
                fused_results.append((person_id, fused_score, person_name, person_quality))
        
        # Sort by fused score and return top candidates
        fused_results.sort(key=lambda x: x[1], reverse=True)
        max_candidates = getattr(config, 'NUCLEUS_MAX_CANDIDATES', 5)
        return fused_results[:max_candidates]
    
    def _update_tracking_history(self, track_id: str, gait_embedding: Optional[np.ndarray],
                               face_embedding: Optional[np.ndarray],
                               matches: List, quality: float):
        """Update tracking history for future temporal consistency checks"""
        if track_id not in self.sequence_history:
            self.sequence_history[track_id] = []
        
        # Store current identification
        history_entry = {
            'gait_embedding': gait_embedding.copy() if gait_embedding is not None else None,
            'face_embedding': face_embedding.copy() if face_embedding is not None else None,
            'quality': quality,
            'best_match': matches[0] if matches else None,
            'all_matches': matches,
            'timestamp': len(self.sequence_history[track_id])
        }
        
        self.sequence_history[track_id].append(history_entry)
        
        # Keep only recent history (last 10 sequences)
        if len(self.sequence_history[track_id]) > 10:
            self.sequence_history[track_id] = self.sequence_history[track_id][-10:]
    
    def get_identification_confidence(self, track_id: str) -> float:
        """Get confidence score for current track identification"""
        if track_id not in self.sequence_history:
            return 0.0
        
        history = self.sequence_history[track_id]
        if not history:
            return 0.0
        
        # Calculate confidence based on consistency and quality
        recent_matches = [h['best_match'] for h in history[-3:] if h['best_match']]
        if not recent_matches:
            return 0.0
        
        # Check consistency
        person_ids = [match[0] for match in recent_matches]
        most_common = max(set(person_ids), key=person_ids.count)
        consistency = person_ids.count(most_common) / len(person_ids)
        
        # Average quality and similarity
        avg_quality = np.mean([h['quality'] for h in history[-3:]])
        avg_similarity = np.mean([h['best_match'][1] for h in history[-3:] if h['best_match']])
        
        # Combined confidence
        confidence = (consistency * 0.4 + avg_quality * 0.3 + avg_similarity * 0.3)
        return confidence
    
    def should_accept_identification(self, track_id: str, threshold: float = 0.6) -> bool:
        """Determine if identification is reliable enough to accept"""
        confidence = self.get_identification_confidence(track_id)
        return confidence >= threshold
    
    def _get_cross_camera_fusion_weights(self, face_matches: List[Tuple], 
                                       gait_matches: List[Tuple],
                                       is_cross_camera: bool = False) -> Tuple[float, float]:
        """
        Get fusion weights adapted for cross-camera scenarios
        
        Args:
            face_matches: Face recognition matches
            gait_matches: Gait recognition matches  
            is_cross_camera: Whether this is a cross-camera scenario
            
        Returns:
            Tuple of (face_weight, gait_weight)
        """
        if is_cross_camera:
            # Use cross-camera specific weights
            base_face_weight = getattr(config, 'CROSS_CAMERA_FACE_WEIGHT', 0.8)
            base_gait_weight = getattr(config, 'CROSS_CAMERA_GAIT_WEIGHT', 0.2)
        else:
            # Use same-camera weights
            base_face_weight = getattr(config, 'SAME_CAMERA_FACE_WEIGHT', 0.7)
            base_gait_weight = getattr(config, 'SAME_CAMERA_GAIT_WEIGHT', 0.3)
        
        # Adaptive weighting based on match quality
        face_quality = np.mean([match[1] for match in face_matches]) if face_matches else 0.0
        gait_quality = np.mean([match[1] for match in gait_matches]) if gait_matches else 0.0
        
        # Boost weights based on relative quality
        if face_quality > 0 and gait_quality > 0:
            total_quality = face_quality + gait_quality
            quality_face_weight = (face_quality / total_quality) * 0.3
            quality_gait_weight = (gait_quality / total_quality) * 0.3
            
            adaptive_face_weight = base_face_weight * 0.7 + quality_face_weight
            adaptive_gait_weight = base_gait_weight * 0.7 + quality_gait_weight
            
            # Normalize
            total_weight = adaptive_face_weight + adaptive_gait_weight
            if total_weight > 0:
                adaptive_face_weight /= total_weight
                adaptive_gait_weight /= total_weight
        else:
            adaptive_face_weight = base_face_weight
            adaptive_gait_weight = base_gait_weight
        
        return adaptive_face_weight, adaptive_gait_weight
