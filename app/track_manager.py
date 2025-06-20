"""
Track management for the gait recognition system
"""

from utils.device import vprint

class TrackManager:
    """Manages track data and lifecycle"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.track_silhouettes = {}
        self.track_embeddings = {}
        self.track_quality_history = {}
        self.track_face_embeddings = {}
        self.track_identities = {}
        self.track_timings = {}
        
        # Processing parameters
        self.MIN_FRAMES_FOR_PROCESSING = 15
        self.QUALITY_CHECK_INTERVAL = 10
        self.RECOGNITION_INTERVAL = 25
        self.MAX_SILHOUETTES_PER_TRACK = 60
        
    def update_tracks(self, silhouette_sequences, face_embeddings, active_track_ids, frame_count):
        """
        Update tracks with new silhouettes and face embeddings
        
        Args:
            silhouette_sequences: Dictionary of track_id -> silhouette list
            face_embeddings: Dictionary of track_id -> face embedding
            active_track_ids: Set of active track IDs
            frame_count: Current frame count
        """
        # Update track silhouettes
        self._update_track_silhouettes(silhouette_sequences, frame_count)
        
        # Update track face embeddings
        self._update_track_face_embeddings(face_embeddings)
        
        # Clean up old tracks
        self._cleanup_inactive_tracks(active_track_ids)
        
    def _update_track_silhouettes(self, silhouette_sequences, frame_count):
        """Update track silhouettes with new sequences"""
        for track_id, sequence in silhouette_sequences.items():
            if not sequence:
                continue
                
            # Initialize track data if new
            if track_id not in self.track_silhouettes:
                self.track_silhouettes[track_id] = []
                self.track_quality_history[track_id] = []
                self.track_timings[track_id] = {'first_seen': frame_count, 'last_seen': frame_count}
            else:
                # Update last seen frame
                self.track_timings[track_id]['last_seen'] = frame_count
            
            # Add new silhouettes
            self.track_silhouettes[track_id].extend(sequence)
            
            # Limit silhouette history to prevent memory issues
            if len(self.track_silhouettes[track_id]) > self.MAX_SILHOUETTES_PER_TRACK:
                self.track_silhouettes[track_id] = self.track_silhouettes[track_id][-self.MAX_SILHOUETTES_PER_TRACK:]
    
    def _update_track_face_embeddings(self, face_embeddings):
        """Update track face embeddings"""
        for track_id, face_embedding in face_embeddings.items():
            if face_embedding is None:
                continue
                
            # Store face embedding
            self.track_face_embeddings[track_id] = face_embedding
                
    def _cleanup_inactive_tracks(self, active_track_ids):
        """Remove data for inactive tracks to save memory"""
        for track_id in list(self.track_silhouettes.keys()):
            if track_id not in active_track_ids:
                # Clean up silhouettes and quality history
                if track_id in self.track_silhouettes:
                    del self.track_silhouettes[track_id]
                if track_id in self.track_quality_history:
                    del self.track_quality_history[track_id]
                if track_id in self.track_face_embeddings:
                    del self.track_face_embeddings[track_id]
                # Keep track timings and identities for conflict detection
                
    def get_tracks_ready_for_processing(self):
        """Get track IDs that are ready for quality assessment and recognition"""
        ready_tracks = []
        for track_id, silhouettes in self.track_silhouettes.items():
            # Check if track has enough frames for processing
            if len(silhouettes) < self.MIN_FRAMES_FOR_PROCESSING:
                continue
                
            # Check if track is due for quality assessment
            if len(silhouettes) % self.QUALITY_CHECK_INTERVAL == 0:
                ready_tracks.append(track_id)
                
        return ready_tracks
        
    def add_quality_assessment(self, track_id, quality_result):
        """Add quality assessment result for a track"""
        if track_id in self.track_quality_history:
            self.track_quality_history[track_id].append(quality_result)
            
            # Log quality assessment
            vprint(f"Track {track_id}: Quality={quality_result['overall_score']:.3f} "
                  f"({quality_result['quality_level']}) - {len(self.track_silhouettes[track_id])} frames")
    
    def get_track_silhouettes(self, track_id):
        """Get silhouettes for a track"""
        return self.track_silhouettes.get(track_id, [])
        
    def get_track_face_embedding(self, track_id):
        """Get face embedding for a track"""
        return self.track_face_embeddings.get(track_id, None)
        
    def set_track_identity(self, track_id, identity_info):
        """Set the identity for a track"""
        self.track_identities[track_id] = identity_info