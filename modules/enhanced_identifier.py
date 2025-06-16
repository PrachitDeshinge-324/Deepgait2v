"""
Enhanced Person Identification for CCTV Gait Recognition
Implements multiple strategies to improve accuracy from 2/5 to higher rates
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from utils.database import PersonEmbeddingDatabase
import config

class CCTVGaitIdentifier:
    """Enhanced identification system for CCTV gait recognition"""
    
    def __init__(self, database: PersonEmbeddingDatabase):
        self.database = database
        self.sequence_history = {}  # Track multiple sequences per person
        self.confidence_scores = {}  # Track confidence over time
        
    def enhanced_identification(self, track_id: str, embedding: np.ndarray, 
                              sequence_quality: float = 0.5) -> List[Tuple]:
        """
        Enhanced identification using multiple strategies for CCTV scenarios
        
        Args:
            track_id: Tracking ID
            embedding: Gait embedding
            sequence_quality: Quality score of the sequence
            
        Returns:
            List of (person_id, confidence, method) tuples
        """
        
        # Strategy 1: Standard nucleus sampling with enhanced parameters
        standard_matches = self.database.identify_person_nucleus(
            embedding,
            top_p=config.NUCLEUS_TOP_P,
            min_candidates=1,
            max_candidates=5,
            close_sim_threshold=config.NUCLEUS_CLOSE_SIM_THRESHOLD,
            amplification_factor=config.NUCLEUS_AMPLIFICATION_FACTOR,
            quality_weight=config.NUCLEUS_QUALITY_WEIGHT,
            enhanced_ranking=config.NUCLEUS_ENHANCED_RANKING
        )
        
        # Strategy 2: Temporal consistency check
        temporal_matches = self._temporal_consistency_check(track_id, embedding)
        
        # Strategy 3: Multi-sequence ensemble if available
        ensemble_matches = self._ensemble_identification(track_id, embedding)
        
        # Strategy 4: Quality-weighted fusion
        final_matches = self._fuse_identification_results(
            standard_matches, temporal_matches, ensemble_matches, sequence_quality
        )
        
        # Update tracking history
        self._update_tracking_history(track_id, embedding, final_matches, sequence_quality)
        
        return final_matches
    
    def _temporal_consistency_check(self, track_id: str, embedding: np.ndarray) -> List[Tuple]:
        """Check consistency with previous identifications for this track"""
        if track_id not in self.sequence_history:
            return []
        
        history = self.sequence_history[track_id]
        if len(history) < 2:
            return []
        
        # Get recent embeddings (last 3-5 sequences)
        recent_embeddings = [h['embedding'] for h in history[-3:]]
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
    
    def _ensemble_identification(self, track_id: str, embedding: np.ndarray) -> List[Tuple]:
        """Use ensemble of multiple embeddings if available"""
        if track_id not in self.sequence_history:
            return []
        
        history = self.sequence_history[track_id]
        if len(history) < 2:
            return []
        
        # Collect recent high-quality embeddings
        recent_embeddings = []
        for h in history[-5:]:  # Last 5 sequences
            if h.get('quality', 0) > 0.5:  # Only high-quality sequences
                recent_embeddings.append(h['embedding'])
        
        if len(recent_embeddings) < 2:
            return []
        
        # Add current embedding
        recent_embeddings.append(embedding)
        
        # Use ensemble identification from database
        try:
            ensemble_results = self.database.identify_person_ensemble(
                recent_embeddings,
                top_k=3,
                threshold=config.IDENTIFICATION_THRESHOLD,
                ensemble_method='weighted_average'
            )
            return ensemble_results
        except:
            return []
    
    def _fuse_identification_results(self, standard: List, temporal: List, 
                                   ensemble: List, quality: float) -> List[Tuple]:
        """Fuse results from multiple identification strategies"""
        
        # Weight different strategies based on their reliability
        weights = {
            'standard': 0.4,
            'temporal': 0.3 if temporal else 0.0,
            'ensemble': 0.3 if ensemble else 0.0
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Collect all person IDs mentioned
        all_person_ids = set()
        for matches in [standard, temporal, ensemble]:
            for match in matches:
                if match:  # Check if match is not None/empty
                    all_person_ids.add(match[0])
        
        # Calculate fused scores for each person
        fused_results = []
        for person_id in all_person_ids:
            fused_score = 0.0
            person_name = None
            person_quality = 0.0
            
            # Standard matches
            for match in standard:
                if match[0] == person_id:
                    fused_score += weights['standard'] * match[1]
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
            
            # Ensemble matches
            for match in ensemble:
                if match[0] == person_id:
                    fused_score += weights['ensemble'] * match[1]
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
        return fused_results[:config.NUCLEUS_MAX_CANDIDATES]
    
    def _update_tracking_history(self, track_id: str, embedding: np.ndarray, 
                               matches: List, quality: float):
        """Update tracking history for future temporal consistency checks"""
        if track_id not in self.sequence_history:
            self.sequence_history[track_id] = []
        
        # Store current identification
        history_entry = {
            'embedding': embedding,
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
