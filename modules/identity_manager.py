"""
Identity Management System for Gait Recognition
Handles temporal consistency, confidence decay, and identity assignment
"""

import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

class IdentityManager:
    """Manages person identities with temporal consistency and confidence decay"""
    
    def __init__(self, config=None):
        """
        Initialize identity manager
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or self._get_default_config()
        
        # Identity tracking: track_id -> identity info
        self.track_identities = {}
        
        # Identity history for temporal consistency: track_id -> deque of (identity, confidence, timestamp)
        self.identity_history = defaultdict(lambda: deque(maxlen=self.config['history_window']))
        
        # Confidence decay tracking: track_id -> last_update_time
        self.last_updates = {}
        
        # Identity switches tracking for debugging
        self.identity_switches = defaultdict(int)
        
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self):
        """Default configuration for identity management"""
        return {
            'history_window': 10,           # Number of previous identifications to keep
            'confidence_threshold': 0.6,    # Minimum confidence for identity assignment
            'majority_vote_threshold': 0.6, # Threshold for majority vote
            'confidence_decay_rate': 0.95,  # Per-second confidence decay rate
            'max_no_detection_time': 5.0,   # Maximum time (seconds) without detection before reset
            'temporal_consistency_weight': 0.7,  # Weight for temporal consistency vs current detection
            'min_consistency_votes': 3,     # Minimum votes needed for temporal consistency
        }
    
    def update_identity(self, track_id, person_id, confidence, person_name=None):
        """
        Update identity for a track with temporal consistency
        
        Args:
            track_id: Track identifier
            person_id: Identified person ID
            confidence: Confidence score of identification
            person_name: Optional person name
            
        Returns:
            dict: Final identity decision with confidence
        """
        current_time = datetime.now()
        
        # Apply confidence decay if this track has been seen before
        if track_id in self.last_updates:
            time_diff = (current_time - self.last_updates[track_id]).total_seconds()
            if time_diff > self.config['max_no_detection_time']:
                # Reset identity if too much time has passed
                self._reset_track_identity(track_id)
            else:
                # Apply confidence decay to existing identity
                self._apply_confidence_decay(track_id, time_diff)
        
        # Add current identification to history
        self.identity_history[track_id].append((person_id, confidence, current_time))
        
        # Calculate final identity using temporal consistency
        final_identity = self._calculate_temporal_identity(track_id)
        
        # Update tracking info
        self.track_identities[track_id] = final_identity
        self.last_updates[track_id] = current_time
        
        # Log identity switches for debugging
        if track_id in self.track_identities and final_identity['person_id'] != person_id:
            self.identity_switches[track_id] += 1
            self.logger.debug(f"Identity switch for track {track_id}: {person_id} -> {final_identity['person_id']}")
        
        return final_identity
    
    def _calculate_temporal_identity(self, track_id):
        """
        Calculate final identity using temporal consistency and majority vote
        
        Args:
            track_id: Track identifier
            
        Returns:
            dict: Identity decision with confidence
        """
        history = self.identity_history[track_id]
        
        if not history:
            return {'person_id': None, 'confidence': 0.0, 'method': 'no_history'}
        
        # If we have only one identification, use it if confidence is high enough
        if len(history) == 1:
            person_id, confidence, timestamp = history[0]
            if confidence >= self.config['confidence_threshold']:
                return {
                    'person_id': person_id,
                    'confidence': confidence,
                    'method': 'single_detection',
                    'person_name': None
                }
            else:
                return {'person_id': None, 'confidence': confidence, 'method': 'low_confidence'}
        
        # Calculate majority vote with confidence weighting
        person_votes = defaultdict(list)  # person_id -> list of (confidence, recency_weight)
        
        current_time = datetime.now()
        for i, (person_id, confidence, timestamp) in enumerate(history):
            # Calculate recency weight (more recent = higher weight)
            time_diff = (current_time - timestamp).total_seconds()
            recency_weight = np.exp(-time_diff / 10.0)  # 10 second half-life
            
            person_votes[person_id].append((confidence, recency_weight))
        
        # Calculate weighted scores for each person
        person_scores = {}
        for person_id, votes in person_votes.items():
            # Weighted average of confidence scores
            total_weight = sum(conf * recency for conf, recency in votes)
            total_recency = sum(recency for conf, recency in votes)
            
            if total_recency > 0:
                weighted_confidence = total_weight / total_recency
                vote_strength = len(votes) / len(history)  # Proportion of votes
                
                # Combine confidence and vote strength
                person_scores[person_id] = {
                    'confidence': weighted_confidence,
                    'vote_strength': vote_strength,
                    'combined_score': weighted_confidence * (0.7 + 0.3 * vote_strength)
                }
        
        if not person_scores:
            return {'person_id': None, 'confidence': 0.0, 'method': 'no_valid_votes'}
        
        # Select the person with highest combined score
        best_person_id = max(person_scores.keys(), key=lambda pid: person_scores[pid]['combined_score'])
        best_score = person_scores[best_person_id]
        
        # Apply temporal consistency threshold
        if (best_score['combined_score'] >= self.config['majority_vote_threshold'] and
            best_score['vote_strength'] >= self.config['min_consistency_votes'] / len(history)):
            
            return {
                'person_id': best_person_id,
                'confidence': best_score['confidence'],
                'vote_strength': best_score['vote_strength'],
                'combined_score': best_score['combined_score'],
                'method': 'temporal_consistency',
                'person_name': None
            }
        else:
            return {
                'person_id': None,
                'confidence': best_score['confidence'],
                'method': 'insufficient_consistency'
            }
    
    def _apply_confidence_decay(self, track_id, time_diff):
        """Apply confidence decay to existing identity"""
        if track_id not in self.track_identities:
            return
            
        decay_factor = self.config['confidence_decay_rate'] ** time_diff
        current_identity = self.track_identities[track_id]
        
        if 'confidence' in current_identity:
            current_identity['confidence'] *= decay_factor
    
    def _reset_track_identity(self, track_id):
        """Reset identity for a track that has been inactive too long"""
        if track_id in self.track_identities:
            del self.track_identities[track_id]
        if track_id in self.identity_history:
            self.identity_history[track_id].clear()
        if track_id in self.last_updates:
            del self.last_updates[track_id]
    
    def get_identity(self, track_id):
        """
        Get current identity for a track
        
        Args:
            track_id: Track identifier
            
        Returns:
            dict: Identity info or None if no identity
        """
        return self.track_identities.get(track_id)
    
    def get_identity_stability(self, track_id):
        """
        Get identity stability metrics for a track
        
        Args:
            track_id: Track identifier
            
        Returns:
            dict: Stability metrics
        """
        if track_id not in self.identity_history:
            return {'stability': 0.0, 'switches': 0, 'history_length': 0}
        
        history = self.identity_history[track_id]
        switches = self.identity_switches[track_id]
        
        # Calculate stability as inverse of switch rate
        stability = 1.0 / (1.0 + switches / max(len(history), 1))
        
        return {
            'stability': stability,
            'switches': switches,
            'history_length': len(history)
        }
    
    def cleanup_old_tracks(self, active_track_ids):
        """
        Clean up identities for tracks that are no longer active
        
        Args:
            active_track_ids: Set of currently active track IDs
        """
        current_time = datetime.now()
        tracks_to_remove = []
        
        for track_id in list(self.track_identities.keys()):
            if track_id not in active_track_ids:
                # Check if this track has been inactive for too long
                if track_id in self.last_updates:
                    time_diff = (current_time - self.last_updates[track_id]).total_seconds()
                    if time_diff > self.config['max_no_detection_time']:
                        tracks_to_remove.append(track_id)
                else:
                    tracks_to_remove.append(track_id)
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            self._reset_track_identity(track_id)
    
    def get_statistics(self):
        """Get identity management statistics"""
        total_tracks = len(self.track_identities)
        identified_tracks = sum(1 for identity in self.track_identities.values() 
                              if identity.get('person_id') is not None)
        total_switches = sum(self.identity_switches.values())
        
        return {
            'total_tracks': total_tracks,
            'identified_tracks': identified_tracks,
            'identification_rate': identified_tracks / max(total_tracks, 1),
            'total_identity_switches': total_switches,
            'average_switches_per_track': total_switches / max(total_tracks, 1)
        }
